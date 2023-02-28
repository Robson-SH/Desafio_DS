from kedro.pipeline import Pipeline, node, pipeline
from .nodes import transforming_data, predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=transforming_data,
            inputs=["train", "test"],
            outputs=["train_fixed", "test_fixed"],
            name="transforming_data"
        ),
        node(
            func=predict,
            inputs=["train_fixed", "test_fixed"],
            outputs="output",
            name="predict"
        )])