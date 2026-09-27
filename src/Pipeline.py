from src.PipelineLink import PipelineLink


class Pipeline:
    """Own and connect an ordered callback pipeline.

    ``PipelineLink`` remains the processing abstraction. ``Pipeline`` is only
    a lightweight container around an ordered set of links so a configured
    callback graph can be passed around as one object.

    Configuration and hashing deliberately remain implemented by
    ``PipelineLink``. The container delegates those calls to its current tail,
    preserving the existing recursive ``previous``-link behaviour unchanged.
    """

    def __init__(
        self,
        links=None,
    ):
        self._links = []

        if links is not None:
            for link in links:
                self.append(link)

    @property
    def links(self):
        return tuple(
            self._links
        )

    @property
    def head(self):
        if not self._links:
            return None

        return self._links[0]

    @property
    def tail(self):
        if not self._links:
            return None

        return self._links[-1]

    def append(
        self,
        link,
    ):
        """Append one link and connect it to the current tail."""

        if not isinstance(
            link,
            PipelineLink,
        ):
            raise TypeError(
                "Pipeline can only contain PipelineLink instances"
            )

        if self.tail is not None:
            self.tail.register_callback(
                link
            )

        self._links.append(
            link
        )

        return link

    def get_link(
        self,
        link_type,
    ):
        """Return the unique link matching ``link_type``."""

        matches = [
            link
            for link in self._links
            if isinstance(
                link,
                link_type,
            )
        ]

        if not matches:
            raise LookupError(
                "Pipeline does not contain a "
                f"{link_type.__name__}"
            )

        if len(matches) > 1:
            raise LookupError(
                "Pipeline contains multiple "
                f"{link_type.__name__} links"
            )

        return matches[0]

    def next_audio(
        self,
        audio,
    ):
        """Send one audio item into an externally driven pipeline."""

        self._require_head().next_audio(
            audio
        )

    def run(
        self,
    ):
        """Run a source-driven pipeline whose head exposes ``stream()``."""

        head = self._require_head()
        stream = getattr(
            head,
            "stream",
            None,
        )

        if stream is None:
            raise TypeError(
                "Pipeline head does not provide stream()"
            )

        return stream()

    def get_config(
        self,
    ):
        return self._require_tail().get_config()

    def get_config_json(
        self,
    ):
        return self._require_tail().get_config_json()

    def get_config_hash(
        self,
    ):
        return self._require_tail().get_config_hash()

    def _require_head(
        self,
    ):
        if self.head is None:
            raise RuntimeError(
                "Cannot use an empty Pipeline"
            )

        return self.head

    def _require_tail(
        self,
    ):
        if self.tail is None:
            raise RuntimeError(
                "Cannot inspect the configuration of an empty Pipeline"
            )

        return self.tail
