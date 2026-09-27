"""Full-dataset embedding precomputation entry point."""

from src.PipelineDefinitions import get_pipeline_definitions
from src.PipelineFactory import PipelineFactory


def main():
    """Run the canonical embedding pipeline over the complete data source."""

    definition = get_pipeline_definitions()[0]
    pipeline = PipelineFactory().build(definition)

    print(
        pipeline.get_config_json()
    )

    pipeline.run()


if __name__ == "__main__":
    main()
