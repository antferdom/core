from famapy.core.operations import Operation
from famapy.core.models.variability_model import VariabilityModel


class Operation3(Operation):
    def execute(self, model: VariabilityModel) -> 'Operation':
        return Operation3()

    def get_result(self):
        return ''
