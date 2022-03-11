import inspect
import logging
from queue import Queue
from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any, Iterator, Optional, Type

from famapy.core.config import PLUGIN_PATHS
from famapy.core.exceptions import OperationNotFound
from famapy.core.exceptions import TransformationNotFound
from famapy.core.models import VariabilityModel
from famapy.core.operations import Operation
from famapy.core.plugins import (
    Operations,
    Plugin,
    Plugins
)
from famapy.core.transformations import Transformation
from famapy.core.transformations.text_to_model import TextToModel
from famapy.core.transformations.model_to_model import ModelToModel


LOGGER = logging.getLogger('discover')


GraphBFS = dict[Type[ModelToModel], list[Type[ModelToModel]]]


def filter_modules_from_plugin_paths() -> list[ModuleType]:
    results: list[ModuleType] = []
    for path in PLUGIN_PATHS:
        try:
            module: ModuleType = import_module(path)
            results.append(module)
        except ModuleNotFoundError:
            LOGGER.exception('ModuleNotFoundError %s', path)
    return results


def filter_m2m(
    m2m_list: list[Type[ModelToModel]],
    input_extension: str,
) -> Iterator[Type[ModelToModel]]:
    return filter(lambda x: x.get_source_extension() == input_extension, m2m_list)


def shortest_way_transformation(
    m2m_transformations: list[Type[ModelToModel]],
    input_extension: str,
    output_extensions: list[str],
) -> list[Type[ModelToModel]]:
    """Use BFS (Breadth First Search) algorithm modified."""

    parent = None
    queue: Queue[Type[ModelToModel]] = Queue()
    visited = set()
    graph: GraphBFS = {}

    # first loop search solution and/or first candidates
    for m2m in filter_m2m(m2m_transformations, input_extension):
        if m2m.get_destination_extension() in output_extensions:
            return [m2m]

        graph[m2m] = []
        queue.put(m2m)
        visited.add(m2m)

    # loop search solution
    solution_found = None
    while 1:
        if queue.empty():
            break
        parent = queue.get()

        for m2m in filter_m2m(m2m_transformations, parent.get_destination_extension()):
            if m2m in visited:
                continue

            graph[parent].append(m2m)
            if m2m not in graph:
                graph[m2m] = []

            queue.put(m2m)
            visited.add(m2m)
            if m2m.get_destination_extension() in output_extensions:
                solution_found = m2m
                break

    return calculate_solution_from_graph(graph, solution_found, input_extension)


def calculate_solution_from_graph(
    graph: GraphBFS,
    solution_found: Type[ModelToModel],
    input_extension: str,
) -> list[Type[ModelToModel]]:

    if solution_found is None:
        raise NotImplementedError("Way to execute operation not found")

    solution = [solution_found]
    while solution[-1].get_source_extension() != input_extension:
        for key, values in graph.items():
            if solution[-1] in values:
                solution.append(key)
                break

    solution.reverse()
    return solution


class DiscoverMetamodels:
    def __init__(self) -> None:
        self.module_paths = filter_modules_from_plugin_paths()
        self.plugins: Plugins = self.discover()

    def search_classes(self, module: ModuleType) -> list[Any]:
        classes = []
        for _, file_name, ispkg in iter_modules(
            module.__path__, module.__name__ + '.'
        ):
            if ispkg:
                classes += self.search_classes(import_module(file_name))
            else:
                _file = import_module(file_name)
                classes += inspect.getmembers(_file, inspect.isclass)
        return classes

    def discover(self) -> Plugins:
        plugins = Plugins()
        for pkg in self.module_paths:
            for _, plugin_name, ispkg in iter_modules(
                pkg.__path__, pkg.__name__ + '.'
            ):
                if not ispkg:
                    continue
                module = import_module(plugin_name)
                plugin = Plugin(module=module)

                classes = self.search_classes(module)

                for _, _class in classes:
                    if not _class.__module__.startswith(module.__package__):
                        continue  # Exclude modules not in current package
                    inherit = _class.mro()

                    if Operation in inherit:
                        plugin.append_operation(_class)
                    elif Transformation in inherit:
                        plugin.append_transformations(_class)
                    elif VariabilityModel in inherit:
                        plugin.variability_model = _class
                plugins.append(plugin)
        return plugins

    def reload(self) -> None:
        self.plugins = self.discover()

    def get_operations(self) -> list[Type[Operation]]:
        """ Get the operations for all modules """
        operations: list[Type[Operation]] = []
        for plugin in self.plugins:
            operations += plugin.operations
        return operations

    def get_name_operations(self) -> list[str]:
        operations = []
        for operation in self.get_operations():
            operations.append(operation.__name__)
            base = operation.__base__.__name__
            if base != 'ABC':
                operations.append(base)

        return operations

    def get_transformations(self) -> list[Type[Transformation]]:
        """ Get transformations for all modules """
        transformations: list[Type[Transformation]] = []
        for plugin in self.plugins:
            transformations += plugin.transformations
        return transformations

    def get_transformations_t2m(self) -> list[Type[TextToModel]]:
        """ Get t2m transformations for all modules """

        transformations: list[Type[TextToModel]] = []
        for plugin in self.plugins:
            transformations += [
                t for t in plugin.transformations if issubclass(t, TextToModel)
            ]
        return transformations

    def get_transformations_m2m(self) -> list[Type[ModelToModel]]:
        """ Get m2m transformations for all modules """

        transformations: list[Type[ModelToModel]] = []
        for plugin in self.plugins:
            transformations += [
                t for t in plugin.transformations if issubclass(t, ModelToModel)
            ]
        return transformations

    def get_operations_by_plugin(self, plugin_name: str) -> Operations:
        return self.plugins.get_operations_by_plugin_name(plugin_name)

    def get_plugins_with_operation(self, operation_name: str) -> list[Plugin]:
        return [
            plugin for plugin in self.plugins
            if operation_name in self.get_name_operations_by_plugin(plugin.name)
        ]

    def get_name_operations_by_plugin(self, plugin_name: str) -> list[str]:
        operations = []
        for operation in self.get_operations_by_plugin(plugin_name):
            operations.append(operation.__name__)
            base = operation.__base__.__name__
            if base != 'ABC':
                operations.append(base)

        return operations

    def get_variability_models(self) -> list[VariabilityModel]:
        return self.plugins.get_variability_models()

    def get_plugins(self) -> list[str]:
        return self.plugins.get_plugin_names()

    def use_transformation_m2t(self, src: VariabilityModel, dst: str) -> str:
        plugin = self.plugins.get_plugin_by_variability_model(src)
        return plugin.use_transformation_m2t(src, dst)

    def use_transformation_t2m(self, src: str, dst: str) -> VariabilityModel:
        plugin = self.plugins.get_plugin_by_extension(dst)
        return plugin.use_transformation_t2m(src)

    def use_transformation_m2m(self, src: VariabilityModel, dst: str) -> VariabilityModel:
        plugin = self.plugins.get_plugin_by_extension(dst)
        return plugin.use_transformation_m2m(src, dst)

    def use_operation(self, src: VariabilityModel, operation: str) -> Operation:
        plugin = self.plugins.get_plugin_by_variability_model(src)
        return plugin.use_operation(operation, src)

    def use_operation_from_file(
        self,
        operation_name: str,
        file: str,
        plugin_name: Optional[str] = None,
    ) -> Any:

        if operation_name not in self.get_name_operations():
            raise OperationNotFound()

        if plugin_name is not None:
            plugin = self.plugins.get_plugin_by_name(plugin_name)
            vm_temp = plugin.use_transformation_t2m(file)
        else:
            vm_temp = self.__transform_to_model_from_file(file)
            plugin = self.plugins.get_plugin_by_extension(vm_temp.get_extension())

            if operation_name not in self.get_name_operations_by_plugin(plugin.name):
                transformation_way = self.__search_transformation_way(
                    plugin, operation_name
                )

                for m2m in transformation_way:
                    vm_temp = m2m(vm_temp).transform()

        operation = self.use_operation(src=vm_temp, operation=operation_name)
        return operation.get_result()

    def __transform_to_model_from_file(self, file: str) -> VariabilityModel:
        t2m_transformations = self.get_transformations_t2m()
        extension = file.split('.')[-1]
        t2m_filters = filter(
            lambda t2m: t2m.get_source_extension() == extension,
            t2m_transformations
        )

        t2m = next(t2m_filters, None)
        if t2m is None:
            raise TransformationNotFound()

        return t2m(file).transform()

    def __search_transformation_way(
        self,
        plugin: Plugin,
        operation_name: str,
    ) -> list[Type[ModelToModel]]:
        """
        Search way to reach plugin with operation_name using m2m transformations
        """

        plugins_with_operation = self.get_plugins_with_operation(
            operation_name)
        m2m_transformations = self.get_transformations_m2m()

        input_extension = plugin.get_extension()
        output_extensions = [p.get_extension() for p in plugins_with_operation]
        return shortest_way_transformation(
            m2m_transformations, input_extension, output_extensions
        )
