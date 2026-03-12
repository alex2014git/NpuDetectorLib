#include "pipeline/npu_pipeline_graph.hpp"
#include "pipeline/npu_pipeline_node.hpp"
#include <queue>
#include <stdexcept>

namespace npu_pipeline {

// Add node to graph
void PipelineGraph::addNode(std::shared_ptr<PipelineNode> node) {
    if (!node) return;
    _nodes[node->getName()] = std::move(node);
}

// Remove node from graph
void PipelineGraph::removeNode(const std::string& node_id) {
    _nodes.erase(node_id);

    // Remove associated edges
    _edges.erase(node_id);
    for (auto& [from, edges] : _edges) {
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [&node_id](const PipelineEdge& e) { return e.to_node == node_id; }),
            edges.end()
        );
    }

    // Remove from reverse edges
    _reverse_edges.erase(node_id);
    for (auto& [to, edges] : _reverse_edges) {
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [&node_id](const PipelineEdge& e) { return e.from_node == node_id; }),
            edges.end()
        );
    }
}

// Add edge between nodes
void PipelineGraph::addEdge(const PipelineEdge& edge) {
    _edges[edge.from_node].push_back(edge);
    _reverse_edges[edge.to_node].push_back(edge);
}

// Remove edge
void PipelineGraph::removeEdge(const std::string& from_node, const std::string& to_node) {
    auto it = _edges.find(from_node);
    if (it != _edges.end()) {
        it->second.erase(
            std::remove_if(it->second.begin(), it->second.end(),
                [&to_node](const PipelineEdge& e) { return e.to_node == to_node; }),
            it->second.end()
        );
    }

    auto rev_it = _reverse_edges.find(to_node);
    if (rev_it != _reverse_edges.end()) {
        rev_it->second.erase(
            std::remove_if(rev_it->second.begin(), rev_it->second.end(),
                [&from_node](const PipelineEdge& e) { return e.from_node == from_node; }),
            rev_it->second.end()
        );
    }
}

// Get node by ID
std::shared_ptr<PipelineNode> PipelineGraph::getNode(const std::string& node_id) const {
    auto it = _nodes.find(node_id);
    if (it != _nodes.end()) {
        return it->second;
    }
    return nullptr;
}

// Check if node exists
bool PipelineGraph::hasNode(const std::string& node_id) const {
    return _nodes.find(node_id) != _nodes.end();
}

// Get all nodes
std::vector<std::shared_ptr<PipelineNode>> PipelineGraph::getAllNodes() const {
    std::vector<std::shared_ptr<PipelineNode>> result;
    result.reserve(_nodes.size());
    for (const auto& [id, node] : _nodes) {
        result.push_back(node);
    }
    return result;
}

// Get input nodes (no incoming edges)
std::vector<std::shared_ptr<PipelineNode>> PipelineGraph::getInputNodes() const {
    std::vector<std::shared_ptr<PipelineNode>> result;
    for (const auto& [id, node] : _nodes) {
        if (_reverse_edges.find(id) == _reverse_edges.end() ||
            _reverse_edges.at(id).empty()) {
            result.push_back(node);
        }
    }
    return result;
}

// Get output nodes (no outgoing edges)
std::vector<std::shared_ptr<PipelineNode>> PipelineGraph::getOutputNodes() const {
    std::vector<std::shared_ptr<PipelineNode>> result;
    for (const auto& [id, node] : _nodes) {
        if (_edges.find(id) == _edges.end() ||
            _edges.at(id).empty()) {
            result.push_back(node);
        }
    }
    return result;
}

// Get downstream nodes
std::vector<std::string> PipelineGraph::getDownstreamNodes(const std::string& node_id) const {
    std::vector<std::string> result;
    auto it = _edges.find(node_id);
    if (it != _edges.end()) {
        for (const auto& edge : it->second) {
            result.push_back(edge.to_node);
        }
    }
    return result;
}

// Get upstream nodes
std::vector<std::string> PipelineGraph::getUpstreamNodes(const std::string& node_id) const {
    std::vector<std::string> result;
    auto it = _reverse_edges.find(node_id);
    if (it != _reverse_edges.end()) {
        for (const auto& edge : it->second) {
            result.push_back(edge.from_node);
        }
    }
    return result;
}

// Get edges from node
std::vector<PipelineEdge> PipelineGraph::getOutputEdges(const std::string& node_id) const {
    auto it = _edges.find(node_id);
    if (it != _edges.end()) {
        return it->second;
    }
    return {};
}

// Get edges to node
std::vector<PipelineEdge> PipelineGraph::getInputEdges(const std::string& node_id) const {
    auto it = _reverse_edges.find(node_id);
    if (it != _reverse_edges.end()) {
        return it->second;
    }
    return {};
}

// Topological sort using Kahn's algorithm
std::vector<std::string> PipelineGraph::topologicalSort() const {
    std::vector<std::string> result;
    std::unordered_map<std::string, int> in_degree;

    // Calculate in-degrees
    for (const auto& [id, node] : _nodes) {
        in_degree[id] = 0;
    }
    for (const auto& [from, edges] : _edges) {
        for (const auto& edge : edges) {
            in_degree[edge.to_node]++;
        }
    }

    // Queue nodes with no dependencies
    std::queue<std::string> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(id);
        }
    }

    // Process nodes
    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();
        result.push_back(current);

        auto it = _edges.find(current);
        if (it != _edges.end()) {
            for (const auto& edge : it->second) {
                in_degree[edge.to_node]--;
                if (in_degree[edge.to_node] == 0) {
                    queue.push(edge.to_node);
                }
            }
        }
    }

    // Check for cycles
    if (result.size() != _nodes.size()) {
        // Cycle detected - return partial result
    }

    return result;
}

// Validate graph
bool PipelineGraph::validate(std::string& error_msg) const {
    // Check for empty graph
    if (_nodes.empty()) {
        error_msg = "Graph is empty";
        return false;
    }

    // Check for cycles using DFS
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> rec_stack;

    for (const auto& [id, node] : _nodes) {
        if (visited.find(id) == visited.end()) {
            if (hasCycleDFS(id, visited, rec_stack)) {
                error_msg = "Graph contains a cycle";
                return false;
            }
        }
    }

    // Check for input nodes
    auto inputs = getInputNodes();
    if (inputs.empty()) {
        error_msg = "Graph has no input nodes";
        return false;
    }

    // Check that all referenced nodes exist
    for (const auto& [from, edges] : _edges) {
        if (!hasNode(from)) {
            error_msg = "Edge references non-existent node: " + from;
            return false;
        }
        for (const auto& edge : edges) {
            if (!hasNode(edge.to_node)) {
                error_msg = "Edge references non-existent node: " + edge.to_node;
                return false;
            }
        }
    }

    return true;
}

// DFS for cycle detection
bool PipelineGraph::hasCycleDFS(const std::string& node,
                                 std::unordered_set<std::string>& visited,
                                 std::unordered_set<std::string>& rec_stack) const {
    visited.insert(node);
    rec_stack.insert(node);

    auto it = _edges.find(node);
    if (it != _edges.end()) {
        for (const auto& edge : it->second) {
            if (visited.find(edge.to_node) == visited.end()) {
                if (hasCycleDFS(edge.to_node, visited, rec_stack)) {
                    return true;
                }
            } else if (rec_stack.find(edge.to_node) != rec_stack.end()) {
                return true;
            }
        }
    }

    rec_stack.erase(node);
    return false;
}

// Clear graph
void PipelineGraph::clear() {
    _nodes.clear();
    _edges.clear();
    _reverse_edges.clear();
}

// Check if edge exists
bool PipelineGraph::hasEdge(const std::string& from_node, const std::string& to_node) const {
    auto it = _edges.find(from_node);
    if (it != _edges.end()) {
        for (const auto& edge : it->second) {
            if (edge.to_node == to_node) {
                return true;
            }
        }
    }
    return false;
}

// Get edge
PipelineEdge PipelineGraph::getEdge(const std::string& from_node,
                                     const std::string& to_node) const {
    auto it = _edges.find(from_node);
    if (it != _edges.end()) {
        for (const auto& edge : it->second) {
            if (edge.to_node == to_node) {
                return edge;
            }
        }
    }
    return PipelineEdge(from_node, to_node, PipelineEdge::PASS_THROUGH);
}

// PipelineGraphBuilder implementation
PipelineGraphBuilder& PipelineGraphBuilder::addNpuNode(const std::string& node_id,
                                                        int algorithm_type,
                                                        const std::string& model_config) {
    TempNode temp;
    temp.id = node_id;
    temp.type = "npu";
    temp.algorithm_type = algorithm_type;
    temp.model_config = model_config;
    _temp_nodes.push_back(std::move(temp));
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::addTransformNode(const std::string& node_id,
                                                              TransformNode::TransformOp op) {
    TempNode temp;
    temp.id = node_id;
    temp.type = "transform";
    temp.transform_op = op;
    _temp_nodes.push_back(std::move(temp));
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::addCustomNode(std::shared_ptr<PipelineNode> node) {
    TempNode temp;
    temp.id = node->getName();
    temp.type = "custom";
    temp.node = std::move(node);
    _temp_nodes.push_back(std::move(temp));
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::addEdge(const PipelineEdge& edge) {
    _temp_edges.push_back(edge);
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::connect(const std::string& from, const std::string& to) {
    _temp_edges.emplace_back(from, to, PipelineEdge::PASS_THROUGH);
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::connectCrop(const std::string& from,
                                                         const std::string& to,
                                                         int target_class) {
    _temp_edges.emplace_back(from, to, PipelineEdge::CROP_ROI);
    if (target_class >= 0) {
        _temp_edges.back().params.target_classes = {target_class};
    }
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::connectBatch(const std::string& from,
                                                          const std::string& to,
                                                          size_t batch_size) {
    PipelineEdge edge(from, to, PipelineEdge::BATCH_ACCUMULATE);
    edge.params.batch_size = batch_size;
    _temp_edges.push_back(std::move(edge));
    return *this;
}

PipelineGraphBuilder& PipelineGraphBuilder::connectFilter(const std::string& from,
                                                           const std::string& to,
                                                           int target_class) {
    PipelineEdge edge(from, to, PipelineEdge::FILTER_CLASS);
    edge.params.target_classes = {target_class};
    _temp_edges.push_back(std::move(edge));
    return *this;
}

std::unique_ptr<PipelineGraph> PipelineGraphBuilder::build() {
    auto graph = std::make_unique<PipelineGraph>();

    // Create nodes from temp nodes
    for (const auto& temp : _temp_nodes) {
        if (temp.node) {
            // Custom node (already created)
            graph->addNode(temp.node);
        } else if (temp.type == "npu") {
            // Create NPU inference node from algorithm_type and model_config
            auto npu_node = std::make_shared<NpuInferenceNode>(temp.id, temp.algorithm_type);
            // Load model config (stored in model_config path)
            npu_node->setModelConfig(temp.model_config);
            graph->addNode(npu_node);
        } else if (temp.type == "transform") {
            // Create transform node
            auto transform_node = std::make_shared<TransformNode>(temp.id, temp.transform_op);
            graph->addNode(transform_node);
        }
        // Other types handled similarly
    }

    // Add edges
    for (const auto& edge : _temp_edges) {
        graph->addEdge(edge);
    }

    return graph;
}

bool PipelineGraphBuilder::validate(std::string& error_msg) const {
    // Check for duplicate node IDs
    std::unordered_set<std::string> node_ids;
    for (const auto& temp : _temp_nodes) {
        if (node_ids.find(temp.id) != node_ids.end()) {
            error_msg = "Duplicate node ID: " + temp.id;
            return false;
        }
        node_ids.insert(temp.id);
    }

    // Check that edge references exist
    for (const auto& edge : _temp_edges) {
        if (node_ids.find(edge.from_node) == node_ids.end()) {
            error_msg = "Edge references non-existent node: " + edge.from_node;
            return false;
        }
        if (node_ids.find(edge.to_node) == node_ids.end()) {
            error_msg = "Edge references non-existent node: " + edge.to_node;
            return false;
        }
    }

    return true;
}

} // namespace npu_pipeline
