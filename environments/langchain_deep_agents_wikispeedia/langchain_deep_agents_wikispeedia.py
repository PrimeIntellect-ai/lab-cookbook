from typing import cast

import verifiers.v1 as vf
from pydantic import Field

if __package__:
    from .wiki_graph import WikiGraph, WikiPair, load_wiki_graph
else:
    from wiki_graph import WikiGraph, WikiPair, load_wiki_graph


def system_prompt(allow_go_back: bool = True) -> str:
    backtracking = (
        "Use `wikispeedia_go_back` to undo your last click."
        if allow_go_back
        else "Backtracking is disabled, so choose each link carefully."
    )
    return f"""This game is easy and fun:

You are given two Wikipedia articles. Starting from the first article, reach the second one exclusively by following available links. Use `wikispeedia_click_link` to navigate.
{backtracking}

When you reach the target the tool will say TARGET REACHED. Stop calling tools and reply with a brief confirmation."""


class WikispeediaState(vf.State):
    current_article: str = ""
    path: list[str] = Field(default_factory=list)
    reached_target: bool = False
    invalid_clicks: int = 0


class WikispeediaTask(vf.Task):
    source: str
    target: str
    shortest_path: int
    links_only: bool = False


class WikispeediaToolConfig(vf.ToolsetConfig):
    cache_dir: str | None = None
    allow_go_back: bool = True


class WikispeediaConfig(vf.TasksetConfig):
    cache_dir: str | None = None
    min_path_length: int = 3
    max_path_length: int = 6
    train_size: int = 50_000
    eval_size: int = 1_000
    eval_target_fraction: float = 0.1
    split_seed: int = 0
    links_only: bool = False
    allow_go_back: bool = True
    efficiency_weight: float = 0.0
    stratify_path_length: bool = True
    tools: WikispeediaToolConfig = WikispeediaToolConfig()


class WikispeediaToolset(vf.Toolset[WikispeediaToolConfig, WikispeediaState]):
    TOOL_PREFIX = "wikispeedia"

    async def setup(self) -> None:
        self.wiki = load_wiki_graph(self.config.cache_dir)

    async def setup_task(self, task: WikispeediaTask) -> None:
        self.source = task.source
        self.target = task.target
        self.links_only = task.links_only

    def _ensure_state(self) -> None:
        if not self.state.current_article:
            self.state.current_article = self.source
            self.state.path = [self.source]

    def format_article(self, article: str, links_only: bool = False) -> str:
        links = self.wiki.get_links(article)
        links_str = ", ".join(links) if links else "(no outgoing links)"
        if links_only:
            return f"# {article}\n\nAvailable links: {links_str}"
        text = self.wiki.get_text(article)
        return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"

    @vf.tool
    async def click_link(self, article: str) -> str:
        """Navigate to a linked Wikipedia article."""
        self._ensure_state()
        available = self.wiki.get_links(self.state.current_article)
        normalized = self.wiki.normalize_name(article)
        if normalized is None or normalized not in available:
            self.state.invalid_clicks += 1
            avail_str = ", ".join(available) if available else "(none)"
            return (
                f"'{article}' is not a valid link from '{self.state.current_article}'.\n"
                f"Available links: {avail_str}"
            )
        self.state.current_article = normalized
        self.state.path.append(normalized)
        if normalized == self.target:
            self.state.reached_target = True
            return (
                f"TARGET REACHED: {normalized}\n\n"
                "You successfully navigated to the target. Stop calling tools and reply briefly."
            )
        return self.format_article(normalized, links_only=self.links_only)

    @vf.tool
    async def go_back(self) -> str:
        """Undo the last click_link and return to the previous article."""
        self._ensure_state()
        if not self.config.allow_go_back:
            return "Backtracking is disabled for this task."
        if len(self.state.path) <= 1:
            return "You are already at the starting article. Cannot go back."
        self.state.path.pop()
        self.state.current_article = self.state.path[-1]
        return self.format_article(self.state.current_article, links_only=self.links_only)


class WikispeediaTaskset(vf.Taskset[WikispeediaTask, WikispeediaConfig, WikispeediaState]):
    def wiki(self) -> WikiGraph:
        wiki_graph = getattr(self, "_wiki_graph", None)
        if wiki_graph is None:
            wiki_graph = load_wiki_graph(self.config.cache_dir)
            self._wiki_graph = wiki_graph
        return wiki_graph

    def format_article(self, article: str, links_only: bool = False) -> str:
        links = self.wiki().get_links(article)
        links_str = ", ".join(links) if links else "(no outgoing links)"
        if links_only:
            return f"# {article}\n\nAvailable links: {links_str}"
        text = self.wiki().get_text(article)
        return f"# {article}\n\n{text}\n\n---\nAvailable links: {links_str}"

    def split_pairs(self) -> tuple[list[WikiPair], list[WikiPair]]:
        return self.wiki().split_pairs(
            train_size=self.config.train_size,
            eval_size=self.config.eval_size,
            min_dist=self.config.min_path_length,
            max_dist=self.config.max_path_length,
            eval_target_fraction=self.config.eval_target_fraction,
            seed=self.config.split_seed,
            stratify=self.config.stratify_path_length,
        )

    def load_tasks(self) -> list[WikispeediaTask]:
        train, _ = self.split_pairs()
        tasks: list[WikispeediaTask] = []
        for idx, (source, target, dist) in enumerate(train):
            starting = self.format_article(source, links_only=self.config.links_only)
            prompt = f"Your mission: {source} >> {target}\n\nStarting article:\n\n{starting}"
            tasks.append(
                WikispeediaTask(
                    idx=idx,
                    name=f"{source}->{target}",
                    prompt=prompt,
                    system_prompt=system_prompt(self.config.allow_go_back),
                    source=source,
                    target=target,
                    shortest_path=dist,
                    links_only=self.config.links_only,
                )
            )
        return tasks

    def tools(self, task: WikispeediaTask) -> list[vf.Toolset]:
        _ = task
        tool_config = self.config.tools.model_copy(
            update={
                "cache_dir": self.config.cache_dir,
                "allow_go_back": self.config.allow_go_back,
            }
        )
        return [cast(vf.Toolset, WikispeediaToolset(tool_config))]

    @vf.stop
    async def reached_target_stop(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> bool:
        return trace.state.reached_target

    @vf.reward(weight=1.0)
    async def reached_target(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> float:
        return float(trace.state.reached_target)

    @vf.metric
    async def path_efficiency(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> float:
        if not trace.state.reached_target:
            return 0.0
        actual = max(len(trace.state.path) - 1, 1)
        return min(1.0, trace.task.shortest_path / actual)

    @vf.reward(weight=1.0)
    async def path_efficiency_reward(
        self, trace: vf.Trace[WikispeediaTask, WikispeediaState]
    ) -> float:
        return self.config.efficiency_weight * await self.path_efficiency(trace)

    @vf.metric
    async def path_length(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> float:
        return float(max(len(trace.state.path) - 1, 0))

    @vf.metric
    async def shortest_path(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> float:
        return float(trace.task.shortest_path)

    @vf.metric
    async def invalid_link_rate(self, trace: vf.Trace[WikispeediaTask, WikispeediaState]) -> float:
        clicks = max(len(trace.state.path) - 1 + trace.state.invalid_clicks, 0)
        return trace.state.invalid_clicks / clicks if clicks else 0.0


if __name__ == "__main__":
    WikispeediaToolset.run()


__all__ = ["WikispeediaTaskset"]
