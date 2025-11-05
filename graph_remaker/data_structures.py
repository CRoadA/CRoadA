from typing import Self

"""
Coordinates in grid (row, col)
"""
GridPoint = tuple[int, int]

class BorderNode:
    value: GridPoint
    _parent: Self
    _children: list[Self]

    def __init__(self, parent: Self, value: GridPoint):
        self._parent = parent
        self.value = value
        self._children = []

    def get_parent(self):
        return self._parent
    
    def get_children(self):
        return self._children.copy()
    
    def get_leftmost_leaf(self):
        if not self._children:
            return self
        return self._children[0].get_leftmost_leaf()

    def appendChild(self, child: Self):
        self._children.append(child)

    def copy_subtree(self, parent: Self = None) -> Self:
        """Copy node with all its decendants.
        Args:
            parent (BorderNode): Node, which the copy should have as a parent.
        """
        copy = BorderNode(parent, self.value)
        copy._children += [child.copy_subtree(copy) for child in self._children]



class StreetBorder:
    _root: BorderNode

    def __init__(self, root: GridPoint = None):
        if GridPoint == None:
            self._root = None
            return
        
        self._root = BorderNode(None, root)

    def _getChildNode(self, node: BorderNode, searched_point: GridPoint):
        if node == None:
            return None
        
        if node.value == searched_point:
            return node
        
        for child in node._children:
            searched = self._getChildNode(child)
            if searched != None:
                return searched
            
        return None

    def getNode(self, point: GridPoint):
        return self._getChildNode(point)
    
    def get_leftmost_leaf(self):
        if self._root == None:
            return None
        
        return self._root.get_leftmost_leaf()
    
    def appendChild(self, parent: GridPoint, child: GridPoint):
        parent_node = self._getChildNode(parent)
        parent_node.children.append(BorderNode(parent, child))

    def copy(self):
        """Shallow copy."""
        new_root = self._root.copy_subtree(None)
        return StreetBorder(new_root)


    def reroot(self, point: GridPoint) -> Self:
        """Reorganize tree, to make given point a root.
        
        Args:
            point (Point): value of node, which is going to become a root.

        Returns:
            StreetBorder: Shallow copy of object with reorganized structure (nodes are new, but values are not copied).
        """

        source = self.getNode(point)
        assert source != None, f"given point is not an element of the StreetBorder"
        new_border = StreetBorder(point)
        target = new_border._root
        # copy all decendants
        target._children += [child_node.copy_subtree(target) for child_node in source._children]

        while source._parent != None: # unless the root of source tree is reached
            previous_source = source
            source = source._parent
            new_target = BorderNode(target, source.value)
            target.appendChild(new_target)
            target = new_target
            # copy all not copied yet
            target._children += [child_node.copy_subtree(target) for child_node in source._children if child_node.value != previous_source.value]
        # points on the other side of border root are just elements of root's subtree, so at this point the whole tree is already copied
        return new_border
    
    def merge(self, merged_border: Self, merging_point: GridPoint, inplace:bool = False):
        """Merge two StreetBorders via given point.
        Args:
            merged_border (StreetBorder): Second StreetBorder to be merged.
            merging_point (Point): Common point indicating, where the StreetBorders need to be merged.
        Returns:
            StreetBorder: If inplace == True, shallow copy with merged StreetBorders. Otherwise the modified object, on which the method is invoked. In both cases root is taken from the object, on which the method is invoked.
        """
        assert self.getNode(merging_point) != None, "first given StreetBorder does not contain given merging point"
        assert merged_border.getNode(merging_point) != None, "second given StreetBorder does not contain given merging point"
        
        if inplace:
            result = self
        else:
            result = self.copy()
        appended = merged_border.reroot(merging_point)
        result.getNode(merging_point)._children += appended._root._children
        return result
    
    def _subtree_to_list(self, node: BorderNode):
        children = node.get_children()
        if not children:
            return [node.value]
        result = []
        for child in children:
            result += child
        return result
    
    def to_list(self):
        return self._subtree_to_list(self._root)
