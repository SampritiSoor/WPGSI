# distutils: language = c++
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, log2
from cython.parallel import prange
from collections import deque
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import heapq

# Import Cython libraries

from libc.math cimport sqrt
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from cython.parallel import prange
from libc.string cimport memcpy
import random
import string
from collections import defaultdict
from scipy.spatial.distance import cosine


cdef float get_min(const cnp.float32_t[:, :] img, int x, int y, int k) nogil:
    """Computes minimum value in a kxk window around (x, y)."""
    cdef int i, j, half_k = k // 2
    cdef float min_val = img[x, y]

    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            min_val = min(min_val, img[x + i, y + j])

    return min_val

cdef float get_max(const cnp.float32_t[:, :] img, int x, int y, int k) nogil:
    """Computes maximum value in a kxk window around (x, y)."""
    cdef int i, j, half_k = k // 2
    cdef float max_val = img[x, y]

    for i in range(-half_k, half_k + 1):
        for j in range(-half_k, half_k + 1):
            max_val = max(max_val, img[x + i, y + j])

    return max_val

cpdef cnp.ndarray[cnp.float32_t, ndim=2] float_erosion(cnp.ndarray[cnp.float32_t, ndim=2] image, int kernel_size=3):
    """Performs grayscale erosion (minimum filter) on a floating-point image."""
    cdef int h = image.shape[0], w = image.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] output = np.zeros((h, w), dtype=np.float32)
    
    cdef int x, y
    cdef int half_k = kernel_size // 2

    cdef cnp.float32_t[:, :] img_view = image  # Use memoryview for fast access
    cdef cnp.float32_t[:, :] out_view = output

    with nogil:
        for x in prange(half_k, h - half_k):
            for y in range(half_k, w - half_k):
                out_view[x, y] = get_min(img_view, x, y, kernel_size)

    return output

cpdef cnp.ndarray[cnp.float32_t, ndim=2] float_dilation(cnp.ndarray[cnp.float32_t, ndim=2] image, int kernel_size=3):
    """Performs grayscale dilation (maximum filter) on a floating-point image."""
    cdef int h = image.shape[0], w = image.shape[1]
    cdef cnp.ndarray[cnp.float32_t, ndim=2] output = np.zeros((h, w), dtype=np.float32)
    
    cdef int x, y
    cdef int half_k = kernel_size // 2

    cdef cnp.float32_t[:, :] img_view = image  # Use memoryview for fast access
    cdef cnp.float32_t[:, :] out_view = output

    with nogil:
        for x in prange(half_k, h - half_k):
            for y in range(half_k, w - half_k):
                out_view[x, y] = get_max(img_view, x, y, kernel_size)

    return output

cpdef cnp.ndarray[cnp.float32_t, ndim=2] float_opening(cnp.ndarray[cnp.float32_t, ndim=2] image, int kernel_size=3):
    """Performs grayscale opening (erosion followed by dilation)."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2] eroded = float_erosion(image, kernel_size)
    return float_dilation(eroded, kernel_size)

cpdef cnp.ndarray[cnp.float32_t, ndim=2] float_closing(cnp.ndarray[cnp.float32_t, ndim=2] image, int kernel_size=3):
    """Performs grayscale closing (dilation followed by erosion)."""
    cdef cnp.ndarray[cnp.float32_t, ndim=2] dilated = float_dilation(image, kernel_size)
    return float_erosion(dilated, kernel_size)



cdef class PriorityQueue:
    """Efficient Priority Queue implemented with a Min-Heap."""

    cdef dict queues  # Stores priority values mapped to lists (deques)
    cdef int total_length  # Total number of elements
    cdef list min_heap  # Min-heap to track priority values

    def __cinit__(self):
        """Initializes an empty PriorityQueue."""
        self.queues = {}
        self.total_length = 0
        self.min_heap = []

    cpdef void push(self, double priority, object element):
        """Push an element into the Priority Queue with a specific priority value."""
        if priority not in self.queues:
            self.queues[priority] = deque()
            heapq.heappush(self.min_heap, priority)

        self.queues[priority].append(element)
        self.total_length += 1

    cpdef tuple pop(self):
        """Pop the first element from the queue with the lowest priority value."""
        if self.total_length == 0:
            raise IndexError("Pop from empty PriorityQueue")

        cdef double min_priority
        cdef object element

        # Remove stale priority values from the heap
        while self.min_heap and self.min_heap[0] not in self.queues:
            heapq.heappop(self.min_heap)

        if not self.min_heap:
            raise IndexError("Pop from empty PriorityQueue")

        min_priority = self.min_heap[0]
        element = self.queues[min_priority].popleft()
        self.total_length -= 1

        # Remove the priority entry if no elements remain
        if not self.queues[min_priority]:
            del self.queues[min_priority]
            heapq.heappop(self.min_heap)

        return min_priority, element

    cpdef int length(self):
        """Returns the total count of elements in the Priority Queue."""
        return self.total_length

    cpdef void display(self):
        """Displays the contents of the Priority Queue."""
        print("PriorityQueue Contents:")
        for priority in sorted(self.queues.keys()):
            print(f"Priority {priority}: {list(self.queues[priority])}")

cdef class OrderedQueue:
    """Efficient Ordered Priority Queue with Min/Max heap functionality."""
    cdef list queue
    cdef dict entry_finder
    cdef int counter
    cdef bint descending

    def __cinit__(self, bint descending=False):
        self.queue = []
        self.entry_finder = {}
        self.counter = 0
        self.descending = descending

    cdef void _heapify(self):
        heapq.heapify(self.queue)

    cpdef void push(self, double order_value, object element):
        """Push an element into the queue while maintaining order."""
        cdef double order = -order_value if self.descending else order_value
        heapq.heappush(self.queue, (order, element))
        self.entry_finder[element] = order
        self.counter += 1

    cpdef tuple pop(self):
        """Removes and returns the highest-priority element."""
        cdef double order
        cdef object element

        while self.queue:
            order, element = heapq.heappop(self.queue)
            if element in self.entry_finder:
                del self.entry_finder[element]
                self.counter -= 1
                return element, (-order if self.descending else order)
        raise IndexError("Pop from empty OrderedQueue")

    cpdef void remove_element(self, object element):
        """Deletes an element from the queue."""
        if element not in self.entry_finder:
            raise KeyError(f"Element '{element}' not found in OrderedQueue")
        del self.entry_finder[element]
        self.counter -= 1

    cpdef int length(self):
        """Returns the number of elements in the queue."""
        return self.counter



cdef str get_unique_identifier():
    """Generate a unique identifier as a random string of letters."""
    cdef int identifier_length = 6
    return ''.join(random.choices(string.ascii_letters, k=identifier_length))

cpdef tuple create_graph_from_image(cnp.ndarray[cnp.float32_t, ndim=3] image, int connectivity=4, str metric='self'):
    """Creates a graph from the input image with pixel-based connectivity."""

    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    # Define neighbor offsets
    # Define neighbor offsets manually for each connectivity level
    cdef int neighbor_offsets_4[4][2]
    neighbor_offsets_4[0][0], neighbor_offsets_4[0][1] = -1, 0
    neighbor_offsets_4[1][0], neighbor_offsets_4[1][1] = 1, 0
    neighbor_offsets_4[2][0], neighbor_offsets_4[2][1] = 0, -1
    neighbor_offsets_4[3][0], neighbor_offsets_4[3][1] = 0, 1

    cdef int neighbor_offsets_8[8][2]
    neighbor_offsets_8[:4] = neighbor_offsets_4  # Copy 4-connectivity neighbors
    neighbor_offsets_8[4][0], neighbor_offsets_8[4][1] = -1, -1
    neighbor_offsets_8[5][0], neighbor_offsets_8[5][1] = -1, 1
    neighbor_offsets_8[6][0], neighbor_offsets_8[6][1] = 1, -1
    neighbor_offsets_8[7][0], neighbor_offsets_8[7][1] = 1, 1

    cdef int neighbor_offsets_12[12][2]
    neighbor_offsets_12[:8] = neighbor_offsets_8  # Copy 8-connectivity neighbors
    neighbor_offsets_12[8][0], neighbor_offsets_12[8][1] = -2, 0
    neighbor_offsets_12[9][0], neighbor_offsets_12[9][1] = 2, 0
    neighbor_offsets_12[10][0], neighbor_offsets_12[10][1] = 0, -2
    neighbor_offsets_12[11][0], neighbor_offsets_12[11][1] = 0, 2

    cdef int neighbor_offsets_24[24][2]
    neighbor_offsets_24[:12] = neighbor_offsets_12  # Copy 12-connectivity neighbors
    neighbor_offsets_24[12][0], neighbor_offsets_24[12][1] = -2, -1
    neighbor_offsets_24[13][0], neighbor_offsets_24[13][1] = -2, 1
    neighbor_offsets_24[14][0], neighbor_offsets_24[14][1] = 2, -1
    neighbor_offsets_24[15][0], neighbor_offsets_24[15][1] = 2, 1
    neighbor_offsets_24[16][0], neighbor_offsets_24[16][1] = -1, -2
    neighbor_offsets_24[17][0], neighbor_offsets_24[17][1] = -1, 2
    neighbor_offsets_24[18][0], neighbor_offsets_24[18][1] = 1, -2
    neighbor_offsets_24[19][0], neighbor_offsets_24[19][1] = 1, 2
    neighbor_offsets_24[20][0], neighbor_offsets_24[20][1] = -2, -2
    neighbor_offsets_24[21][0], neighbor_offsets_24[21][1] = -2, 2
    neighbor_offsets_24[22][0], neighbor_offsets_24[22][1] = 2, -2
    neighbor_offsets_24[23][0], neighbor_offsets_24[23][1] = 2, 2


    cdef int (*neighbor_offsets)[2]
    cdef int num_offsets

    if connectivity == 4:
        neighbor_offsets = neighbor_offsets_4
        num_offsets = 4
    elif connectivity == 8:
        neighbor_offsets = neighbor_offsets_8
        num_offsets = 8
    elif connectivity == 12:
        neighbor_offsets = neighbor_offsets_12
        num_offsets = 12
    elif connectivity == 24:
        neighbor_offsets = neighbor_offsets_24
        num_offsets = 24
    else:
        raise ValueError("Connectivity must be 4, 8, 12, or 24")

    # Declare required Cython variables before loops
    cdef dict node_map_cord_to_id = {}
    cdef dict node_map_id_to_cord = {}
    cdef dict id_neighbours = {}

    cdef cnp.ndarray[cnp.float32_t, ndim=2] reconstructed_image = np.zeros((h, w), dtype=np.float32)

    cdef int dx, dy, nx, ny, i
    cdef float node_weight
    cdef cnp.float32_t[:] pixel_value
    cdef list neighbors

    # Generate unique identifiers for each pixel
    for x in range(h):  
        for y in range(w):  
            u = get_unique_identifier()
            node_map_cord_to_id[(x, y)] = u
            node_map_id_to_cord[u] = (x, y)

    # Compute graph properties using `range()` instead of `prange()`
    for x in range(h):  
        for y in range(w):
            pixel_value = image[x, y, :]  # Fix: Explicitly buffer slice

            # Get neighbors within bounds
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    neighbors.append((nx, ny))

            id_neighbours[node_map_cord_to_id[(x, y)]] = [node_map_cord_to_id[n] for n in neighbors]

            # Compute node weight
            if metric == 'self':
                node_weight = 0.3 * pixel_value[0] + 0.6 * pixel_value[1] + 0.1 * pixel_value[2]
            elif metric == 'euclidean':
                node_weight = min([sqrt((pixel_value[0] - image[n[0], n[1], 0]) ** 2 +
                                        (pixel_value[1] - image[n[0], n[1], 1]) ** 2 +
                                        (pixel_value[2] - image[n[0], n[1], 2]) ** 2)
                                   for n in neighbors])
            else:
                raise ValueError("Unsupported metric. Choose 'self' or 'euclidean'.")

            reconstructed_image[x, y] = node_weight

    # Initialize graph
    cdef dict graph = {}

    for x in range(h):
        for y in range(w):
            node_id = node_map_cord_to_id[(x, y)]
            graph[node_id] = {
                'value': image[x, y, :],
                'weight': reconstructed_image[x, y],
                'dist': np.inf,
                'neighbors': id_neighbours[node_id]
            }

    return graph, node_map_id_to_cord, reconstructed_image

def graph_to_image(graph, node_map_id_to_cord, image_shape):
    # Initialize the image with zeros
    h, w = image_shape
    reconstructed_image = np.zeros((h, w), dtype=np.float32)

    # Map node weights back to pixel coordinates
    for node_id, properties in graph.items():
        x, y = node_map_id_to_cord[node_id]
        reconstructed_image[x, y] = properties['weight']

    return reconstructed_image

def eta(arc, fzpair_arcNodes_dict_hierarchy, id_fzPair_dict_hierarchy, level):
    """
    Recursively find all pixels corresponding to an arc in the hierarchy.

    Parameters:
        arc (frozenset): The arc whose pixels are to be retrieved.
        fzpair_arcNodes_dict_hierarchy (list): List of arc dictionaries for each level.
        id_fzPair_dict_hierarchy (list): List of ID-to-flat zone pair mappings for each level.

    Returns:
        set: A set of pixel coordinates corresponding to the input arc.
    """
    # Determine the current level of the arc
    current_level = level

    # Check if the arc exists in the current level
    if current_level < 0 or arc not in fzpair_arcNodes_dict_hierarchy[current_level]:
        raise ValueError("Arc does not exist in the current hierarchy")

    # Initialize a set to store the coordinates of all pixels
    pixel_set = set()

    # Retrieve partition points associated with the arc
    partition_points = fzpair_arcNodes_dict_hierarchy[current_level][arc]
    # print("partition_points",partition_points)
    # Check if we are at the lowest level (level 0)
    if current_level == 0:
        # At level 0, partition points are pixel nodes
        for point in partition_points:
            # print("id_fzPair_dict_hierarchy",id_fzPair_dict_hierarchy)
            # print("point",point)
            pixel_set.add(id_fzPair_dict_hierarchy[0][point])
    else:
        # At higher levels, recursively find the pixels corresponding to each partition point
        for point in partition_points:
            # Get the flat zone pair associated with the point
            fz_pair = id_fzPair_dict_hierarchy[current_level][point]

            # Recursively call eta for the lower level
            lower_level_pixels = eta(fz_pair, fzpair_arcNodes_dict_hierarchy[:current_level], id_fzPair_dict_hierarchy[:current_level], level = current_level-1)
            pixel_set.update(lower_level_pixels)

    return pixel_set

cpdef list get_initial_points(dict graph):
    """Finds initial points using priority queue-based processing."""
    cdef PriorityQueue priorityqueue = PriorityQueue()
    cdef list initial_points = []
    cdef str node
    cdef double weight

    for node, properties in graph.items():
        properties['mark'] = 'undiscovered'
        priorityqueue.push(properties['weight'] + 0.0001, node)

    while priorityqueue.length() > 0:
        priority, p = priorityqueue.pop()
        if graph[p]['mark'] == 'undiscovered' and graph[p]['weight'] < priority:
            graph[p]['mark'] = 'discovered'
            priorityqueue.push(graph[p]['weight'], p)
            initial_points.append(p)
        else:
            for q in graph[p]['neighbors']:
                if graph[q]['weight'] >= graph[p]['weight'] and graph[q]['mark'] == 'undiscovered':
                    graph[q]['mark'] = 'discovered'
                    priorityqueue.push(graph[q]['weight'], q)

    return initial_points



cpdef tuple generate_hierarchical_graph(
    dict fzpair_arcNodes_dict, dict flatzone_nbrArcs_dict, dict flatzone_valueinfo_dict
):
    """
    Generate a node-weighted graph for the next hierarchical level.
    
    Returns:
        - new_graph (dict): Node-weighted graph for the next hierarchical level.
        - fzPair_id_dict (dict): Maps flat zone pairs (frozenset) to node IDs.
        - id_fzPair_dict_this_level (dict): Maps node IDs to flat zone pairs.
    """

    # ✅ Declare dictionaries to store mappings
    cdef dict new_graph = {}
    cdef dict fzPair_id_dict = {}
    cdef dict id_fzPair_dict_this_level = {}

    # ✅ Declare necessary variables
    cdef int i, j, f
    cdef str new_node_id
    cdef float weight
    cdef frozenset fzpair
    cdef list fz, arc_list
    cdef set partition_points
    cdef frozenset arc1, arc2
    cdef str node1, node2
    cdef object flatzone

    # ✅ Step 1: Create nodes in the new graph
    for i, (fzpair, partition_points) in enumerate(fzpair_arcNodes_dict.items()):
        new_node_id = get_unique_identifier()  # Use index as unique node ID
        fzPair_id_dict[fzpair] = new_node_id
        id_fzPair_dict_this_level[new_node_id] = fzpair

        if partition_points:
            fz = list(fzpair)
            weight = sqrt(np.sum(np.square(
                np.array(flatzone_valueinfo_dict[fz[0]]['mean']) - np.array(flatzone_valueinfo_dict[fz[1]]['mean'])
            )))
        else:
            raise ValueError(f"fzpair {fzpair} not found in generate_hierarchical_graph")

        new_graph[new_node_id] = {
            'weight': weight,
            'neighbors': set()
        }

    # ✅ Step 2: Create edges between nodes using `flatzone_nbrArcs_dict`
    for f, (flatzone, arcs) in enumerate(flatzone_nbrArcs_dict.items()):
        arc_list = list(arcs)

        for i in range(len(arc_list)):
            for j in range(i + 1, len(arc_list)):
                arc1, arc2 = arc_list[i], arc_list[j]

                if arc1 in fzPair_id_dict and arc2 in fzPair_id_dict:
                    node1, node2 = fzPair_id_dict[arc1], fzPair_id_dict[arc2]
                    new_graph[node1]['neighbors'].add(node2)
                    new_graph[node2]['neighbors'].add(node1)

    # ✅ Convert `set` to `list` for each node's neighbors
    for node in new_graph:
        new_graph[node]['neighbors'] = list(new_graph[node]['neighbors'])

    return new_graph, id_fzPair_dict_this_level




cpdef tuple get_initial_arcs(dict graph, list initial_points, str metric="euclidean"):
    """
    Initializes arcs in the graph and partitions nodes into flat zones.

    Parameters:
        graph (dict): The graph containing nodes and their properties.
        initial_points (list): A list of starting points.
        metric (str): Distance metric ('euclidean' or 'cosine').

    Returns:
        tuple: (fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict)
    """

    cdef OrderedQueue orderedqueue = OrderedQueue()  # Initialize priority queue

    # Declare dictionaries
    cdef dict fzpair_arcNodes_dict
    cdef dict flatzone_nbrArcs_dict
    cdef dict flatzone_values_dict
    cdef dict fzpair_arcInfo_dict
    cdef dict flatzone_valueinfo_dict

    # Initialize as standard Python dicts
    fzpair_arcNodes_dict = {}
    flatzone_nbrArcs_dict = {}
    flatzone_values_dict = {}
    fzpair_arcInfo_dict = {}
    flatzone_valueinfo_dict = {}

    # Mark all nodes as undiscovered
    cdef str node
    cdef dict properties
    cdef str p, q
    cdef dict p_properties, q_properties
    cdef float dist

    for node, properties in graph.items():
        properties['mark'] = 'undiscovered'
        properties['label'] = None

    # Initialize initial points
    for node in initial_points:
        graph[node]['weight'] = 0
        graph[node]['label'] = node
        orderedqueue.push(0, node)

    # Process the queue
    while orderedqueue.length() > 0:
        p, _ = orderedqueue.pop()
        p_properties = graph[p]

        if p_properties['mark'] == 'visited':
            continue

        p_properties['mark'] = 'visited'

        # Process neighbors
        for q in p_properties['neighbors']:
            q_properties = graph[q]

            if q_properties['mark'] == 'visited':
                continue

            dist = p_properties['weight']

            if metric == "euclidean":
                dist += sqrt(np.sum(np.square(np.array(p_properties['value']) - np.array(q_properties['value']))))
            elif metric == "cosine":
                dist += cosine(np.array(p_properties['value']), np.array(q_properties['value']))

            if q_properties['mark'] == "discovered" and q_properties['label'] == p_properties['label']:
                if dist < q_properties['weight']:
                    orderedqueue.remove_element(q)
                    q_properties['weight'] = dist
                    orderedqueue.push(dist, q)
            elif q_properties['mark'] == "undiscovered":
                q_properties['label'] = p_properties['label']
                q_properties['mark'] = "discovered"
                orderedqueue.push(dist, q)
            else:
                q_properties['label'] = "partition point"
                q_properties['mark'] = "visited"

    # Post-process to create arc and flat zone dictionaries
    cdef str node_id
    cdef list neighbor_labels
    cdef frozenset fzPair_key
    cdef int i, j

    for node_id, properties in graph.items():
        if properties['label'] == "partition point":
            neighbor_labels = [
                graph[n]['label'] for n in properties['neighbors'] if graph[n]['label'] != "partition point"
            ]

            for i in range(len(neighbor_labels)):
                for j in range(i + 1, len(neighbor_labels)):
                    if neighbor_labels[i] != neighbor_labels[j]:
                        fzPair_key = frozenset((neighbor_labels[i], neighbor_labels[j]))

                        if fzPair_key not in fzpair_arcNodes_dict:
                            fzpair_arcNodes_dict[fzPair_key] = set()
                        fzpair_arcNodes_dict[fzPair_key].add(node_id)

            for label in neighbor_labels:
                if label not in flatzone_nbrArcs_dict:
                    flatzone_nbrArcs_dict[label] = set()
                for other_label in neighbor_labels:
                    if label != other_label:
                        fzPair_key = frozenset((label, other_label))
                        flatzone_nbrArcs_dict[label].add(fzPair_key)

    # Compute flat zone values
    for node_id, properties in graph.items():
        if properties['label'] == "partition point":
            continue
        if properties['label'] not in flatzone_values_dict:
            flatzone_values_dict[properties['label']] = []
        flatzone_values_dict[properties['label']].append(properties['value'])

    # Compute flat zone info
    for fz in flatzone_values_dict:
        flatzone_valueinfo_dict[fz] = {
            "count": len(flatzone_values_dict[fz]),
            "mean": np.mean(flatzone_values_dict[fz], axis=0)
        }

    # Compute arc info (ensuring `frozenset` keys work correctly)
    for fp in fzpair_arcNodes_dict:
        fzpair_arcInfo_dict[fp] = {"count": 0, "mean": np.array([0, 0, 0])}

        for node in fzpair_arcNodes_dict[fp]:
            fzpair_arcInfo_dict[fp]["count"] += 1
            fzpair_arcInfo_dict[fp]["mean"] = fzpair_arcInfo_dict[fp]["mean"] + np.array(graph[node]["value"])

        fzpair_arcInfo_dict[fp]["mean"] /= fzpair_arcInfo_dict[fp]["count"]

    return fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict

cpdef tuple watershed_partition(dict graph, dict fzpair_arcInfo_dict_prev, dict flatzone_valueinfo_dict_prev, dict id_to_fzPair, float reintroduce_factor=0):
    """
    Performs watershed partitioning on the graph to segment it into flat zones.

    Parameters:
        graph (dict): The graph containing nodes and their properties.
        fzpair_arcInfo_dict_prev (dict): Previous arc information.
        flatzone_valueinfo_dict_prev (dict): Previous flat zone information.
        id_to_fzPair (dict): Mapping of node IDs to flat zone pairs.
        reintroduce_factor (float): Threshold for reintroducing weak edges.

    Returns:
        tuple: (fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict)
    """

    cdef PriorityQueue priorityqueue = PriorityQueue()  # Initialize priority queue

    # Initialize dictionaries
    cdef dict fzpair_arcNodes_dict = {}
    cdef dict flatzone_nbrArcs_dict = {}
    cdef dict flatzone_nodes_dict = {}

    # Mark all nodes as undiscovered and push into the queue
    cdef str node
    cdef dict properties

    for node, properties in graph.items():
        properties['mark'] = 'undiscovered'
        properties['label'] = None
        priorityqueue.push(np.round(properties['weight'] + 0.0001, 4), node)

    # Process the queue
    cdef str p, q
    cdef dict p_properties, q_properties
    cdef float priority

    while priorityqueue.length() > 0:
        priority, p = priorityqueue.pop()
        p_properties = graph[p]

        if p_properties['mark'] == 'visited':
            continue

        if p_properties['weight'] < priority:
            p_properties['label'] = p
            p_properties['mark'] = 'discovered'
            priorityqueue.push(p_properties['weight'], p)
        else:
            p_properties['mark'] = 'visited'
            for q in p_properties['neighbors']:
                q_properties = graph[q]
                if q_properties['label'] == p_properties['label']:
                    continue
                if q_properties['weight'] >= p_properties['weight']:
                    if q_properties['mark'] == 'undiscovered':
                        q_properties['label'] = p_properties['label']
                        q_properties['mark'] = 'discovered'
                        priorityqueue.push(q_properties['weight'], q)
                    else:
                        q_properties['label'] = 'partition point'
                        q_properties['mark'] = 'visited'

    # Post-process to create arc and flat zone dictionaries
    cdef str node_id
    cdef list neighbor_labels
    cdef frozenset fzPair_key
    cdef int i, j

    for node_id, properties in graph.items():
        if properties['label'] == "partition point":
            neighbor_labels = [
                graph[n]['label'] for n in properties['neighbors'] if graph[n]['label'] != "partition point"
            ]

            for i in range(len(neighbor_labels)):
                for j in range(i + 1, len(neighbor_labels)):
                    if neighbor_labels[i] != neighbor_labels[j]:
                        fzPair_key = frozenset((neighbor_labels[i], neighbor_labels[j]))
                        if fzPair_key not in fzpair_arcNodes_dict:
                            fzpair_arcNodes_dict[fzPair_key] = set()  # Manually initialize
                        fzpair_arcNodes_dict[fzPair_key].add(node_id)

            for label in neighbor_labels:
                if label not in flatzone_nbrArcs_dict:
                    flatzone_nbrArcs_dict[label] = set()
                for other_label in neighbor_labels:
                    if label != other_label:
                        fzPair_key = frozenset((label, other_label))
                        flatzone_nbrArcs_dict[label].add(fzPair_key)

    # Compute flat zone values
    for node_id, properties in graph.items():
        if properties['label'] == "partition point":
            continue
        if properties['label'] not in flatzone_nodes_dict:
            flatzone_nodes_dict[properties['label']] = []
        flatzone_nodes_dict[properties['label']].append(node_id)

    cdef dict flatzone_valueinfo_dict = {
        fz: {"count": 0, "mean": np.array([0, 0, 0])} for fz in flatzone_nodes_dict
    }

    for fz in flatzone_nodes_dict:
        arcs_prevLevel = []
        fzs_prevLevel = set()
        for node in flatzone_nodes_dict[fz]:
            arcs_prevLevel.append(id_to_fzPair[node])
            fz1, fz2 = list(id_to_fzPair[node])
            fzs_prevLevel.add(fz1)
            fzs_prevLevel.add(fz2)

        for arc in arcs_prevLevel:
            flatzone_valueinfo_dict[fz]['count'] += fzpair_arcInfo_dict_prev[arc]['count']
            flatzone_valueinfo_dict[fz]['mean'] = flatzone_valueinfo_dict[fz]['mean'] + fzpair_arcInfo_dict_prev[arc]['mean'] * fzpair_arcInfo_dict_prev[arc]['count']
        for fz_prev in fzs_prevLevel:
            flatzone_valueinfo_dict[fz]['count'] += flatzone_valueinfo_dict_prev[fz_prev]['count']
            flatzone_valueinfo_dict[fz]['mean'] = flatzone_valueinfo_dict[fz]['mean'] + flatzone_valueinfo_dict_prev[fz_prev]['mean'] * flatzone_valueinfo_dict_prev[fz_prev]['count']

        flatzone_valueinfo_dict[fz]['mean'] /= flatzone_valueinfo_dict[fz]['count']

    # Compute arc info
    cdef dict fzpair_arcInfo_dict = {fp: {"count": 0, "mean": np.array([0, 0, 0])} for fp in fzpair_arcNodes_dict}

    for fp in fzpair_arcNodes_dict:
        arcNodes = fzpair_arcNodes_dict[fp]
        arcs_prevLevel = []
        for node in arcNodes:
            arcs_prevLevel.append(id_to_fzPair[node])

        for arc in arcs_prevLevel:
            fzpair_arcInfo_dict[fp]['count'] += fzpair_arcInfo_dict_prev[arc]['count']
            fzpair_arcInfo_dict[fp]['mean'] = fzpair_arcInfo_dict[fp]['mean'] + fzpair_arcInfo_dict_prev[arc]['mean'] * fzpair_arcInfo_dict_prev[arc]['count']
        fzpair_arcInfo_dict[fp]['mean'] = fzpair_arcInfo_dict[fp]['mean'] / fzpair_arcInfo_dict[fp]['count']

    # Reintroduce factor-based adjustments
    if reintroduce_factor:
        count = 0
        for F in flatzone_nodes_dict:
            fzpairs = flatzone_nbrArcs_dict[F]
            weights = [[graph[N]['weight'], fzp] for fzp in fzpairs for N in fzpair_arcNodes_dict[fzp]]
            minWeight, maxWeight = [np.Inf, None], [-np.Inf, None]

            for w in weights:
                if minWeight[0] > w[0]:
                    minWeight = w
                if maxWeight[0] < w[0]:
                    maxWeight = w

            for N in flatzone_nodes_dict[F]:
                if graph[N]['weight'] > reintroduce_factor * minWeight[0]:
                    count += 1
                    fzpair_arcNodes_dict[minWeight[1]].add(N)

        print(count)

    return fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict



cpdef tuple generate_hierarchy(cnp.ndarray[cnp.float32_t, ndim=3] image, int connectivity=8):
    """
    Generates a hierarchical segmentation of the image.

    Parameters:
        image (np.ndarray): Input image.
        connectivity (int): Connectivity (default 8).
        returnImages (bool): Whether to return partition images.

    Returns:
        dict: Hierarchy details.
    """

    # Lists to store hierarchical data
    cdef list partition_pixel_hierarchy = []
    cdef list arcs_hierarchy = []
    cdef list flat_zones_hierarchy = []
    cdef list entropy_hierarchy = []

    cdef list graph_hierarchy = []
    cdef list fzpair_arcNodes_dict_hierarchy = []
    cdef list id_fzPair_dict_hierarchy = []

    # Generate level-0 graph
    cdef dict graph_0, map_id_to_cord_0
    cdef cnp.ndarray[cnp.float32_t, ndim=2] reconstructed_image
    graph_0, map_id_to_cord_0, reconstructed_image = create_graph_from_image(image, connectivity=connectivity, metric='euclidean')

    id_fzPair_dict_hierarchy.append(map_id_to_cord_0)
    graph_hierarchy.append(graph_0)

    cdef int h = image.shape[0]
    cdef int w = image.shape[1]

    cdef list initial_points = get_initial_points(graph_0)
    cdef dict fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict
    fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict = get_initial_arcs(graph_hierarchy[0], initial_points)

    cdef list partitionImages = []
    cdef int level = 0


    # Declare necessary variables before entering the loop
    cdef list coordinates
    cdef dict graph, id_to_fzPair
    cdef cnp.ndarray[cnp.float32_t, ndim=2] w_image = np.ones((h, w), dtype=np.float32)
    cdef float entropy
    
    while True:
        if not fzpair_arcNodes_dict:
            break
    
        fzpair_arcNodes_dict_hierarchy.append(fzpair_arcNodes_dict)
        arcs_hierarchy.append(len(fzpair_arcNodes_dict))
        flat_zones_hierarchy.append(len(flatzone_nbrArcs_dict))
    
        # Compute entropy
        sizes = [fz_nodes['count'] for fz_nodes in flatzone_valueinfo_dict.values()]
        total_size = sum(sizes)
        probs = [size / total_size for size in sizes]
        entropy = -sum([p * log2(p + 1e-9) for p in probs])
        entropy_hierarchy.append(entropy)
    
        # Compute partition pixels
        coordinates = []
        for arc in fzpair_arcNodes_dict:
            coordinates.extend(list(eta(arc, fzpair_arcNodes_dict_hierarchy, id_fzPair_dict_hierarchy, level=level)))
        level += 1
        partition_pixel_hierarchy.append(len(coordinates))
    
        # Create partition image (reuse memory)
        w_image[:] = 1.0  # Reset image instead of allocating new memory
        for c in coordinates:
            w_image[c] = 0
        partitionImages.append(w_image.copy())  # Avoid reference issues
    
        # Generate hierarchical graph
        graph, id_to_fzPair = generate_hierarchical_graph(fzpair_arcNodes_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict)
        graph_hierarchy.append(graph)
        id_fzPair_dict_hierarchy.append(id_to_fzPair)
    
        # Partition the current graph
        fzpair_arcNodes_dict, fzpair_arcInfo_dict, flatzone_nbrArcs_dict, flatzone_valueinfo_dict = watershed_partition(
            graph, fzpair_arcInfo_dict, flatzone_valueinfo_dict, id_to_fzPair, reintroduce_factor=0
        )


    # Compute total nodes (pixels) in the graph
    cdef int total_nodes = h * w

    # Convert lists to NumPy arrays
    arcs_hierarchy_np = np.array(arcs_hierarchy, dtype=np.int32)
    flat_zones_hierarchy_np = np.array(flat_zones_hierarchy, dtype=np.int32)

    # Prepare the hierarchical data dictionary
    cdef dict hierarchy = {
        'partition_pixels': partition_pixel_hierarchy,
        'arcs': arcs_hierarchy_np,
        'flat_zones': flat_zones_hierarchy_np,
        'entropy': entropy_hierarchy,
        'levels': len(arcs_hierarchy),
        'total_nodes': total_nodes
    }

    return hierarchy, partitionImages