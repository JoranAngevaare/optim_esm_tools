import os
import typing

from treelib import Node, Tree


class SyndaViewer:
    """Visualize synda downloads as a tree structure"""

    def __init__(self,
                 base: str = '/nobackup/users/angevaar/synda/data',
                 max_depth: typing.Optional[int] = None,
                 show_files: bool = False,
                 concatenate_folders: bool = True,
                 ):
        """
        Viewer for Synda Folder structure

        :param base: where to start looking
        :param max_depth: maximum recursion depth from the base to show folders
        :param show_files: list files as well as folders
        :param concatenate_folders: concatenate folder names if they contain only one subfolder
        """
        self.base = base
        self.max_depth = max_depth
        self.show_files = show_files
        self.concatenate_folders = concatenate_folders

    def tree(self) -> Tree:
        base = self.base
        tree = Tree()
        tree.create_node(base, base)
        for head, directories, files in os.walk(base):
            if self._skip_deep(head):
                continue
            parent = head
            if self._skip_concatenate(directories, files) <= 1:
                continue

            parent_is_known = parent in tree.nodes.keys()
            look_back = 0
            if not parent_is_known:
                split = self.chopped_path(head)
                for look_back in range(len(split)):
                    if (parent := os.path.join(*(['/'] + split[:look_back + 1]))) in tree.nodes.keys():
                        break

            for sub_dir in directories:
                if look_back:
                    label = os.path.join(*(split[-look_back:] + [sub_dir]))
                else:
                    label = sub_dir
                tree.create_node(label, os.path.join(head, sub_dir), parent=parent)

            if self.show_files and len(files) > 1:
                for file in files:
                    if head not in tree.nodes.keys():
                        # Still have to add the parent directory if it wasn't otherwise added.
                        label = os.path.join(*(split[-look_back:] + [sub_dir]))
                        tree.create_node(label, head, parent=parent)
                    tree.create_node(file, os.path.join(head, file), parent=head)
        return tree

    def _skip_concatenate(self, directories, files) -> bool:
        return self.concatenate_folders and (len(directories) + len(files))

    def _skip_deep(self, head) -> bool:
        return self.max_depth and self.count_depth(head) - self.count_depth(self.base) > self.max_depth

    @staticmethod
    def chopped_path(path) -> list:
        path = os.path.normpath(path)
        return path.split(os.sep)

    def count_depth(self, path) -> int:
        return len(self.chopped_path(path))

    def show(self) -> None:
        self.tree().show()
