from dataclasses import dataclass
from typing import Self

"""
Coordinates in grid (row, col)
"""
GridPoint = tuple[int, int]

MeshPoint = tuple[float, float] # point on Mesh


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

    def _get_child_node(self, node: BorderNode, searched_point: GridPoint) -> BorderNode | None:
        if node == None:
            return None
        
        if node.value == searched_point:
            return node
        
        for child in node._children:
            searched = self._get_child_node(child)
            if searched != None:
                return searched
            
        return None
    
    
    def _get_child_node_depth(self, root_node: BorderNode, searched_node: BorderNode) -> int:
        """Get depth of a child node in a subtree specified by given node.

        Parameters
        ----------
        root_node : BorderNode
            Root of regarded subtree.
        searched_node : BorderNode
            Searched node.

        Returns
        -------
        depth: int
            Depth of searched node node. 0, if root node is searched node. -1, if child node is not present in a subtree.
        """
        if searched_node == root_node:
            return 0
        
        for child in root_node._children:
            depth = self._get_child_node_depth(child, searched_node)
            if depth != -1:
                return depth + 1
        
        return -1
    
    def _calculate_distance_between_nodes(self, root_node: BorderNode, first_node: BorderNode, second_node: BorderNode) -> int:
        """Calculate distance between nodes along the border in a subtree. This is a sum of depths of given points relative to the common ancestor. Does not check, if given nodes exist in the border tree. See calculate_distance_between_points().

        Parameters
        ----------
        root_node: BorderNode
            Root node of regarded subtree.
        first : GridPoint
            First Point.
        second : GridPoint
            Second Point.

        Returns
        -------
        distance: int
            Distance between points. 0, if given points are the same point. -1, if some of the points is not present in the tree.
        """

        # Handle, if one is root
        if first_node == root_node:
            return self._get_child_node_depth(self._root, second_node)
        
        if second_node == root_node:
            return self._get_child_node_depth(self._root, first_node)
        
        # Try to calculate for each child
        for child in root_node._children:
            distance_through_child = self._calculate_distance_between_nodes(child, first_node, second_node)
            if distance_through_child != -1:
                return distance_through_child
        
        # No child has both given nodes as descendants, so calculate it by yourself
        left_depth = self._get_child_node_depth(root_node, first_node)
        if left_depth == -1:
            return - 1
        
        right_depth = self._get_child_node_depth(root_node, second_node)
        if right_depth == -1:
            return - 1
        
        return left_depth + right_depth
    

    def calculate_distance_between_points(self, first: GridPoint, second: GridPoint) -> int:
        """Calculate distance between points along the border. This is a sum of depths of given points relative to the common ancestor.

        Parameters
        ----------
        first : GridPoint
            First Point.
        second : GridPoint
            Second Point.

        Returns
        -------
        distance: int
            Distance between points. 0, if given points are the same point. -1, if some of the points is not present in the tree.
        """

        # Check, if they are present in the border
        first_node = self.getNode(first)
        second_node = self.getNode(second)

        if first_node is None or second_node is None:
            return -1
        
        # Handle equality case
        if first_node == second_node:
            return 0

        return self._calculate_distance_between_nodes(self._root, first_node, second_node)


    def getNode(self, point: GridPoint) -> BorderNode | None:
        return self._get_child_node(point)
    
    def get_leftmost_leaf(self):
        if self._root == None:
            return None
        
        return self._root.get_leftmost_leaf()
    
    def appendChild(self, parent: GridPoint, child: GridPoint):
        parent_node = self._get_child_node(parent)
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
    
    # def _get_minimal_subtree_root(self, current_root: BorderNode, first: GridPoint, second: GridPoint) -> tuple[GridPoint | None, GridPoint | None, GridPoint | None]:
    #     """Helper function for get_minimal_subtree_root, which is invoked on specific node.

    #     Parameters
    #     ----------
    #     current_root : GridPoint
    #         Root node, which subtree needs to be checked.
    #     first : GridPoint
    #         First point.
    #     second : GridPoint
    #         Second point.

    #     Returns
    #     -------
    #     first_node: GridPoint
    #         Node of the first given point. If not None, the problem is already solved.
    #     second_node: GridPoint
    #         Node of the second given point. If None, it is not found in the subtree.
    #     root: GridPoint
    #         Lowest common root of given points. If None, it is not found in the subtree.
    #     """

    #     returned_root, returned_first, returned_second = None, None, None
    #     if current_root.value == first:
    #         returned_first = current_root
    #         # check, if descendant is the searched right node
    #         for child in current_root._children:
    #             _, _, got_second = self._get_minimal_subtree_root(child, current_root.value, second)
    #             if got_second is not None:
    #                 # found solution

    
    # def get_minimal_subtree_root(self, first: GridPoint, second: GridPoint) -> tuple[GridPoint, GridPoint, GridPoint]:
    #     """Find nodes of given points and thier lowest (in terms of tree structure) common ancestor.

    #     Parameters
    #     ----------
    #     first : GridPoint
    #         First point.
    #     second : GridPoint
    #         Second point.

    #     Returns
    #     -------
    #     first_node: GridPoint
    #         Node of the first given point.
    #     second_node: GridPoint
    #         Node of the second given point.
    #     root: GridPoint
    #         Lowest common root of given points.
    #     """

@dataclass
class StreetConflict:
    """Conflict of street with crossroads or grid parrt border.
    Attributes:
        conflict_points (list[GridPoint]): Grid points involved in conflict.
        linestring_points (list[MeshPoint]): Points of street's linestring involved in conflict.
    """
    conflict_points: list[GridPoint]
    linestring_points: list[MeshPoint]

@dataclass
class StreetDiscovery:
    """Discovered data about street during the street discovery process.
    Attributes:
        linestring (list[MeshPoint]): Graph representation of street (part of Mesh). May be empty.
        borders (list[StreetBorder]): Discovered borders of Street.
        conflicts (list[StreetConflict]): Conflicts with Crossroads or grid part border.
    """
    linestring: list[MeshPoint]
    borders: list[StreetBorder]
    conflicts: list[StreetConflict]



@dataclass
class CrossroadDiscovery:
    """Discovered data about street during the street discovery process.
    Attributes:
        points (list[GridPoint]): Points in the interior of the crossroad.
        conflicting_points (list[GridPoint]): Conflicting points of the interior of the crossroad.
        street_juntions (list[tuple[StreetDiscovery, MeshPoint]]): List of adjacentStreetDiscoveries with their junction points (parts of Mesh) binding the StreetDiscovery to the crossroad.
    """
    points: list[GridPoint]
    conflicting_points: list[GridPoint]
    street_juntions: list[tuple[StreetDiscovery, MeshPoint]]