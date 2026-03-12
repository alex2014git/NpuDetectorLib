#ifndef _NPU_PIPELINE_GRAPH_HPP_
#define _NPU_PIPELINE_GRAPH_HPP_

#include "npu_pipeline_types.hpp"
#include "npu_pipeline_edge.hpp"
#include "npu_pipeline_node.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <algorithm>
#include <stdexcept>

namespace npu_pipeline {

// Pipeline graph (DAG)
class PipelineGraph {
public:
    PipelineGraph() = default;
    ~PipelineGraph() = default;

    // Disable copy
    PipelineGraph(const PipelineGraph&) = delete;
    PipelineGraph& operator=(const PipelineGraph&) = delete;

    // Enable move
    PipelineGraph(PipelineGraph&&) = default;
    PipelineGraph& operator=(PipelineGraph&&) = default;

    // Add node to graph
    void addNode(std::shared_ptr<PipelineNode> node);

    // Remove node from graph
    void removeNode(const std::string& node_id);

    // Add edge between nodes
    void addEdge(const PipelineEdge& edge);

    // Remove edge
    void removeEdge(const std::string& from_node, const std::string& to_node);

    // Get node by ID
    std::shared_ptr<PipelineNode> getNode(const std::string& node_id) const;

    // Check if node exists
    bool hasNode(const std::string& node_id) const;

    // Get all nodes
    std::vector<std::shared_ptr<PipelineNode>> getAllNodes() const;

    // Get input nodes (no incoming edges)
    std::vector<std::shared_ptr<PipelineNode>> getInputNodes() const;

    // Get output nodes (no outgoing edges)
    std::vector<std::shared_ptr<PipelineNode>> getOutputNodes() const;

    // Get downstream nodes
    std::vector<std::string> getDownstreamNodes(const std::string& node_id) const;

    // Get upstream nodes
    std::vector<std::string> getUpstreamNodes(const std::string& node_id) const;

    // Get edges from node
    std::vector<PipelineEdge> getOutputEdges(const std::string& node_id) const;

    // Get edges to node
    std::vector<PipelineEdge> getInputEdges(const std::string& node_id) const;

    // Topological sort
    std::vector<std::string> topologicalSort() const;

    // Validate graph (check for cycles, disconnected components, etc.)
    bool validate(std::string& error_msg) const;

    // Check if graph is empty
    bool empty() const { return _nodes.empty(); }

    // Get node count
    size_t size() const { return _nodes.size(); }

    // Clear graph
    void clear();

    // Check if edge exists
    bool hasEdge(const std::string& from_node, const std::string& to_node) const;

    // Get edge
    PipelineEdge getEdge(const std::string& from_node, const std::string& to_node) const;

private:
    // Adjacency list for edges
    std::unordered_map<std::string, std::vector<PipelineEdge>> _edges;

    // Reverse adjacency list
    std::unordered_map<std::string, std::vector<PipelineEdge>> _reverse_edges;

    // Node storage
    std::unordered_map<std::string, std::shared_ptr<PipelineNode>> _nodes;

    // DFS for cycle detection
    bool hasCycleDFS(const std::string& node,
                     std::unordered_set<std::string>& visited,
                     std::unordered_set<std::string>& rec_stack) const;

    // DFS for topological sort
    void topologicalSortDFS(const std::string& node,
                            std::unordered_set<std::string>& visited,
                            std::vector<std::string>& result) const;
};

// Pipeline graph builder for convenient construction
class PipelineGraphBuilder {
public:
    PipelineGraphBuilder() = default;

    // Add node with type
    template<typename T, typename... Args>
    PipelineGraphBuilder& addNode(const std::string& node_id, Args&&... args);

    // Add NPU inference node
    PipelineGraphBuilder& addNpuNode(const std::string& node_id,
                                      int algorithm_type,
                                      const std::string& model_config);

    // Add transform node
    PipelineGraphBuilder& addTransformNode(const std::string& node_id,
                                            TransformNode::TransformOp op);

    // Add custom node
    PipelineGraphBuilder& addCustomNode(std::shared_ptr<PipelineNode> node);

    // Add edge
    PipelineGraphBuilder& addEdge(const PipelineEdge& edge);

    // Convenience edge methods
    PipelineGraphBuilder& connect(const std::string& from, const std::string& to);
    PipelineGraphBuilder& connectCrop(const std::string& from, const std::string& to,
                                       int target_class = -1);
    PipelineGraphBuilder& connectBatch(const std::string& from, const std::string& to,
                                        size_t batch_size);
    PipelineGraphBuilder& connectFilter(const std::string& from, const std::string& to,
                                         int target_class);

    // Build graph
    std::unique_ptr<PipelineGraph> build();

    // Validate during build
    bool validate(std::string& error_msg) const;

    // Get node count
    size_t nodeCount() const { return _temp_nodes.size(); }

    // Get edge count
    size_t edgeCount() const { return _temp_edges.size(); }

private:
    struct TempNode {
        std::string id;
        std::string type;
        std::shared_ptr<PipelineNode> node;
        int algorithm_type = 0;
        std::string model_config;
        TransformNode::TransformOp transform_op = TransformNode::RESIZE;
    };

    std::vector<TempNode> _temp_nodes;
    std::vector<PipelineEdge> _temp_edges;
};

// Template implementation
// Note: This is a forward declaration; implementation needs to be in header or explicitly instantiated
template<typename T, typename... Args>
PipelineGraphBuilder& PipelineGraphBuilder::addNode(const std::string& node_id, Args&&... args) {
    TempNode temp;
    temp.id = node_id;
    temp.node = std::make_shared<T>(node_id, std::forward<Args>(args)...);
    _temp_nodes.push_back(std::move(temp));
    return *this;
}

} // namespace npu_pipeline

#endif // _NPU_PIPELINE_GRAPH_HPP_
