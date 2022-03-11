import pytest

from famapy.core.transformations.model_to_model import ModelToModel
from famapy.core.models.variability_model import VariabilityModel
from famapy.core.discover import DiscoverMetamodels


def factory_model_to_model(src: str, dst: str) -> ModelToModel:
    class M2M(ModelToModel):
        @staticmethod
        def get_source_extension() -> str:
            return src

        @staticmethod
        def get_destination_extension() -> str:
            return dst

        def __init__(self, source_model: VariabilityModel) -> None:
            pass

        def __str__(self):
            return (
                f"{self.get_source_extension()} -> {self.get_destination_extension()}"
            )

        def __repr__(self):
            return str(self)

    return M2M(None)


@pytest.mark.parametrize('output_extensions', [["C"], ["C", "E"]])
def test_transformation_way_shortest_no_solution(output_extensions):
    a_b = factory_model_to_model("A", "B")
    b_c = factory_model_to_model("B", "D")
    m2m_list = [a_b, b_c]
    with pytest.raises(NotImplementedError):
        DiscoverMetamodels()._shortest_way_transformation(m2m_list, "A", output_extensions)


@pytest.mark.parametrize('output_extensions', [["B"], ["B", "C"]])
def test_transformation_way_shortest_easy(output_extensions):
    a_b = factory_model_to_model("A", "B")
    m2m_list = [a_b]
    assert DiscoverMetamodels()._shortest_way_transformation(m2m_list, "A", output_extensions) == [a_b]


@pytest.mark.parametrize('output_extensions', [["C"], ["E", "C"]])
def test_transformation_way_shortest_medium(output_extensions):
    a_b = factory_model_to_model("A", "B")
    b_c = factory_model_to_model("B", "C")
    m2m_list = [a_b, b_c]
    assert DiscoverMetamodels()._shortest_way_transformation(m2m_list, "A", output_extensions) == [
        a_b,
        b_c,
    ]


@pytest.mark.parametrize('output_extensions', [["F"], ["F", "G"]])
def test_transformation_way_shortest_complex(output_extensions):
    a_b = factory_model_to_model("A", "B")
    a_e = factory_model_to_model("A", "E")
    b_c = factory_model_to_model("B", "C")
    b_e = factory_model_to_model("B", "E")
    c_d = factory_model_to_model("C", "D")
    e_d = factory_model_to_model("E", "D")
    e_b = factory_model_to_model("E", "B")
    d_f = factory_model_to_model("D", "F")

    m2m_list = [a_b, a_e, b_c, b_e, c_d, e_d, e_b, d_f]

    assert DiscoverMetamodels()._shortest_way_transformation(m2m_list, "B", output_extensions) == [
        b_c, c_d, d_f
    ]
