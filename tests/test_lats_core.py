import unittest

from langchain_core.messages import AIMessage

from lats import LATSConfig, LATSConfigError, select_leaf, should_continue
from lats.models.node import SearchNode
from lats.models.reflection import Reflection


def _node(score: int, solved: bool = False, parent: SearchNode | None = None) -> SearchNode:
    reflection = Reflection(reflections="test", score=score, found_solution=solved)
    return SearchNode(messages=[AIMessage(content="ok")], reflection=reflection, parent=parent)


class LATSCoreTests(unittest.TestCase):
    def test_config_validation_rejects_invalid_candidates(self) -> None:
        with self.assertRaises(LATSConfigError):
            LATSConfig(n_candidates=0).validate()

    def test_backpropagates_reward_to_ancestors(self) -> None:
        root = _node(score=5)
        child = _node(score=10, parent=root)

        self.assertEqual(root.visits, 2)
        self.assertEqual(child.visits, 1)
        self.assertGreater(root.value, 0.0)

    def test_select_leaf_uses_uct_and_returns_leaf(self) -> None:
        root = _node(score=5)
        left = _node(score=10, parent=root)
        right = _node(score=1, parent=root)
        root.children.extend([left, right])
        left_leaf = _node(score=9, parent=left)
        left.children.append(left_leaf)

        selected = select_leaf(root, exploration_weight=1.0)
        self.assertTrue(selected.is_terminal)

    def test_should_continue_stops_on_solution(self) -> None:
        root = _node(score=5, solved=False)
        solved_child = _node(score=10, solved=True, parent=root)
        root.children.append(solved_child)

        self.assertEqual(should_continue(root, max_depth=5), "__end__")

    def test_should_continue_stops_on_depth(self) -> None:
        root = _node(score=5)
        child = _node(score=5, parent=root)
        root.children.append(child)

        self.assertEqual(should_continue(root, max_depth=2), "__end__")


if __name__ == "__main__":
    unittest.main()
