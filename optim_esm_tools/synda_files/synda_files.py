import os
import typing


class SyndaViewer:
    """Visualize synda downloads as a tree structure."""

    def __init__(
        self,
        base: str = os.path.join(os.environ.get('ST_HOME', 'NO ST HOME?'), 'data'),
        max_depth: typing.Optional[int] = None,
        show_files: bool = False,
        concatenate_folders: bool = True,
        count_files: bool = False,
    ):
        """Viewer for Synda Folder structure.

        :param base: where to start looking
        :param max_depth: maximum recursion depth from the base to show
            folders
        :param show_files: list files as well as folders
        :param concatenate_folders: concatenate folder names if they
            contain only one subfolder
        :param count_files: count files per folder
        """
        self.base = base
        self.max_depth = max_depth
        self.show_files = show_files
        self.concatenate_folders = concatenate_folders
        self.count_files = count_files

    def tree(self):
        from treelib import Tree

        base = self.base
        tree = Tree()
        tree.create_node(base, base)
        last_head = base
        for head, directories, files in os.walk(base):
            if self._skip_deep(head):
                continue
            split = self.chopped_path(head)
            look_back = None
            for look_back in range(len(split) - 1):
                last_head = os.sep.join(split[:-look_back])
                if last_head in tree.nodes.keys():
                    break
            assert look_back is not None
            if last_head in ['', '/']:  # pragma: no cover
                last_head = base

            if self._add_head(directories):
                label = os.path.join(*(split[len(self.chopped_path(last_head)) :]))
                if head not in tree.nodes.keys():
                    if self.count_files and files:
                        label += f' ({len(files)})'
                    tree.create_node(label, head, parent=last_head)

            if self.show_files and len(files):
                for file in files:
                    tree.create_node(file, os.path.join(head, file), parent=head)
        return tree

    def _add_head(self, directories) -> bool:
        return len(directories) != 1 if self.concatenate_folders else True

    def _skip_deep(self, head) -> bool:
        return (
            self.max_depth
            and self.count_depth(head) - self.count_depth(self.base) > self.max_depth
        )  # type: ignore

    @staticmethod
    def chopped_path(path) -> list:
        path = os.path.normpath(path)
        return path.split(os.sep)

    def count_depth(self, path) -> int:
        return len(self.chopped_path(path))

    def show(self) -> None:
        self.tree().show()
