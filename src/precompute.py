"""Full-dataset embedding precomputation entry point."""

from src.PipelineDefinitions import build_embedding_pipeline


def main():
    """Run the canonical embedding pipeline over the complete data source."""

    pipeline = build_embedding_pipeline()

    print(
        pipeline.get_config_json()
    )

    pipeline.run()


if __name__ == "__main__":
    main()
