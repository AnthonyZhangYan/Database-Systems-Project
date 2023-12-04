#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <vector>
#include <queue>
#include <functional>
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <stack>
#include <math.h>
using namespace std;

struct TreeNode;
struct Node;
struct Edge;

const int MAX_DEPTH = 100;
const int DECAY_PARAMETER = 10;
const double ALPHA = 0.02;
const int TAU = 300;

struct MIP {
    vector<Node*> path;
    double propagationProbability;

    MIP() : propagationProbability(0.0) {}

    MIP(const MIP& other) : path(other.path), propagationProbability(other.propagationProbability) {}
};

struct Grid {
    double minX, maxX, minY, maxY;
    unordered_map<int, double> nodeInfluence; // Maps node ID to its influence in this grid

    bool operator==(const Grid& other) const {
        return minX == other.minX && maxX == other.maxX &&
               minY == other.minY && maxY == other.maxY;
    }
};

// Hash function for Grid
namespace std {
    template<>
    struct hash<Grid> {
        size_t operator()(const Grid& grid) const {
            // Combine the hash values of the grid boundaries
            return hash<double>()(grid.minX) ^ (hash<double>()(grid.maxX) << 1) ^
                   (hash<double>()(grid.minY) << 2) ^ (hash<double>()(grid.maxY) << 3);
        }
    };
}

struct AnchorPoint {
    double x, y;
    bool operator==(const AnchorPoint& other) const {
        return x == other.x && y == other.y;
    }
};


namespace std {
    template<>
    struct hash<AnchorPoint> {
        size_t operator()(const AnchorPoint& anchor) const {
            return hash<double>()(anchor.x) ^ (hash<double>()(anchor.y) << 1);
        }
    };
}


struct Node {
    int id;
    double longitude, latitude; // Added longitude and latitude
    unordered_set<Node*> inNeighbors;
    unordered_set<Node*> outNeighbors;
    vector<Edge> outgoingEdges;
    vector<Edge> incomingEdges;
    // For rule 1
    unordered_map<AnchorPoint, double> influenceAtAnchors;
    unordered_map<Grid, double> influenceInGrids;

    // Default constructor
    Node() : id(-1), longitude(0.0), latitude(0.0) {}

    // Parameterized constructor
    Node(int id, double longi, double lati) : id(id), longitude(longi), latitude(lati) {}

    double calculateOutgoingProbability() {
        return outNeighbors.empty() ? 0.0 : 1.0 / outNeighbors.size();
    }
};

Node myNode(1, -73.935242, 40.730610); // Example for creating a Node object

struct Edge {
    Node* from;
    Node* to;
    double probability;

    Edge(Node* fromNode, Node* toNode, double prob) : from(fromNode), to(toNode), probability(prob) {}
};

struct TreeNode  : public std::enable_shared_from_this<TreeNode> {
    shared_ptr<Node> graphNode;
    weak_ptr<TreeNode> parent;
    vector<shared_ptr<TreeNode>> children;

    TreeNode(shared_ptr<Node> graphNode) : graphNode(graphNode) {}

    static shared_ptr<TreeNode> Create(shared_ptr<Node> node) {
        return make_shared<TreeNode>(node);
    }


    void DFS() {
        cout << "Node ID: " << graphNode->id << endl;
        for (auto& child : children) {
            child->DFS();
        }
    }

    vector<shared_ptr<Node>> getAllNodes() {
        vector<shared_ptr<Node>> nodes;
        stack<shared_ptr<TreeNode>> stack;
        stack.push(shared_from_this());
        while (!stack.empty()) {
            auto currentNode = stack.top();
            stack.pop();
            nodes.push_back(currentNode->graphNode);
            for (auto& child : currentNode->children) {
                stack.push(child);
            }
        }
        return nodes;
    }
};




struct Viewpoint {
    double longitude;
    double latitude;

    Viewpoint(double lon, double lat) : longitude(lon), latitude(lat) {}
};


struct CompareProb {
    bool operator()(const pair<double, Node*>& a, const pair<double, Node*>& b) {
        return a.first < b.first;
    }
};

class Graph {
public:
    unordered_map<int, shared_ptr<Node>> nodes;
    double minX, maxX, minY, maxY;

public:
    Graph() = default;

    double assignRandomProbability() {
        // Trivalency model probabilities
        const double probabilities[3] = {0.1, 0.01, 0.001};
        int index = rand() % 3; // Randomly select an index
        return probabilities[index];
    }
    void addEdge(int fromId, int toId, double defaultLongitude, double defaultLatitude) {
        if (nodes.find(fromId) == nodes.end()) {
            nodes[fromId] = make_shared<Node>(fromId, defaultLongitude, defaultLatitude);
        }
        if (nodes.find(toId) == nodes.end()) {
            nodes[toId] = make_shared<Node>(toId, defaultLongitude, defaultLatitude);
        }


        Node* fromNode = nodes[fromId].get();
        Node* toNode = nodes[toId].get();

        fromNode->outNeighbors.insert(toNode);
        toNode->inNeighbors.insert(fromNode);
        //double probability = fromNode->calculateOutgoingProbability();
        double probability = 0.6;
        fromNode->outgoingEdges.emplace_back(fromNode, toNode, probability);
        toNode->incomingEdges.emplace_back(fromNode, toNode, probability);
    }

    void readEdgesFromFile(const string& filename,  double defaultLongitude, double defaultLatitude) {
        ifstream file(filename);
        int fromId, toId;

        while (file >> fromId >> toId) {
            addEdge(fromId, toId, defaultLongitude, defaultLatitude);
        }
    }


    Node* getNodeById(int id) {
        return nodes[id].get();
    }

    vector<Edge> getIncomingEdges(int nodeId) {
        if (nodes.find(nodeId) == nodes.end()) {
            return vector<Edge>();
        }
        return nodes[nodeId]->incomingEdges;
    }

    vector<Edge> getOutgoingEdges(int nodeId) {
        if (nodes.find(nodeId) == nodes.end()) {
            return vector<Edge>();
        }
        return nodes[nodeId]->outgoingEdges;
    }

    double calculateDistance(double long1, double lat1, double long2, double lat2) {
        // Simple Euclidean distance calculation
        return sqrt(pow(long1 - long2, 2) + pow(lat1 - lat2, 2));
    }

    double f(Node* v, pair<double, double> q, double c = DECAY_PARAMETER) {
        // Weight calculation based on distance with decay parameter 'c'
        double distance = calculateDistance(v->longitude, v->latitude, q.first, q.second);
        return exp(-c * distance); // Exponential decay based on distance and decay parameter
    }


    pair<MIP*, unordered_map<Node*, Node*>> CalculateMIP(Node* u, Node* v, pair<double, double> q) {
        unordered_map<Node*, double> MaxProb;
        unordered_map<Node*, Node*> Predecessor;
        priority_queue<pair<double, Node*>, vector<pair<double, Node*>>, CompareProb> PriorityQueue;

        for (auto& pair : nodes) {
            MaxProb[pair.second.get()] = (pair.second.get() == u) ? 1.0 : 0.0;
            Predecessor[pair.second.get()] = nullptr;
        }

        PriorityQueue.push(make_pair(1.0, u));

        while (!PriorityQueue.empty()) {
            Node* CurrentNode = PriorityQueue.top().second;
            PriorityQueue.pop();

            if (CurrentNode == v) break;

            for (Edge& edge : CurrentNode->outgoingEdges) {
                Node* Neighbor = edge.to;
                double NewProb = MaxProb[CurrentNode] * edge.probability * f(Neighbor, q);

                if (NewProb > MaxProb[Neighbor]) {
                    MaxProb[Neighbor] = NewProb;
                    Predecessor[Neighbor] = CurrentNode;
                    PriorityQueue.push(make_pair(NewProb, Neighbor));
                }
            }
        }

        MIP* mip = new MIP();
        Node* CurrentNode = v;
        mip->propagationProbability = MaxProb[v];

        while (CurrentNode != nullptr && CurrentNode != u) {
            mip->path.push_back(CurrentNode);
            CurrentNode = Predecessor[CurrentNode];
        }
        mip->path.push_back(u);
        reverse(mip->path.begin(), mip->path.end());

        return {mip, Predecessor};
    }



    
    shared_ptr<TreeNode> calculateMIIA(int nodeId, double theta, pair<double, double> q) {
        if (nodes.find(nodeId) == nodes.end()) return nullptr;

        shared_ptr<Node> target = nodes[nodeId];
        auto root = make_shared<TreeNode>(target);
        unordered_map<Node*, shared_ptr<TreeNode>> createdNodes;
        createdNodes[target.get()] = root;

        for (auto& pair : nodes) {
            shared_ptr<Node> start = pair.second;
            if (start == target) continue;

            auto mipAndPredecessors = CalculateMIP(start.get(), target.get(), q);
            MIP* mip = mipAndPredecessors.first;

            if (mip->propagationProbability >= theta) {
                shared_ptr<TreeNode> lastAddedNode = root;

                for (auto it = mip->path.rbegin(); it != mip->path.rend(); ++it) {
                    Node* node = *it;
                    if (node == target.get() || createdNodes.find(node) != createdNodes.end()) continue;

                    auto newNode = make_shared<TreeNode>(nodes[node->id]);
                    lastAddedNode->children.push_back(newNode);
                    newNode->parent = lastAddedNode;
                    createdNodes[node] = newNode;
                    lastAddedNode = newNode;
                }
            }
            delete mip;
        }

        return root;
    }


    shared_ptr<TreeNode> calculateMIOA(int nodeId, double theta, pair<double, double> q) {
        if (nodes.find(nodeId) == nodes.end()) return nullptr;

        shared_ptr<Node> source = nodes[nodeId];
        auto root = make_shared<TreeNode>(source);
        unordered_map<Node*, shared_ptr<TreeNode>> createdNodes;
        createdNodes[source.get()] = root;

        for (auto& pair : nodes) {
            shared_ptr<Node> target = pair.second;
            if (source == target) continue;

            auto mipAndPredecessors = CalculateMIP(source.get(), target.get(), q);
            MIP* mip = mipAndPredecessors.first;

            if (mip->propagationProbability >= theta) {
                shared_ptr<TreeNode> lastAddedNode = root;

                for (Node* node : mip->path) {
                    if (node == source.get() || createdNodes.find(node) != createdNodes.end()) continue;

                    auto newNode = make_shared<TreeNode>(nodes[node->id]);
                    lastAddedNode->children.push_back(newNode);
                    newNode->parent = lastAddedNode;
                    createdNodes[node] = newNode;
                    lastAddedNode = newNode;
                }
            }
            delete mip;
        }

        return root;
    }

/*
    //Calculating anchor points
    vector<AnchorPoint> generateAnchorPoints(int numAnchorPoints) {
        vector<AnchorPoint> anchors;

        // Calculate the dimensions of each cell
        int numRows = sqrt(numAnchorPoints);
        int numCols = (numAnchorPoints + numRows - 1) / numRows; // Adjust for rounding

        double cellWidth = (maxX - minX) / numCols;
        double cellHeight = (maxY - minY) / numRows;

        // Generate anchor points at the center of each cell
        for (int row = 0; row < numRows; ++row) {
            for (int col = 0; col < numCols; ++col) {
                double centerX = minX + cellWidth * (col + 0.5);
                double centerY = minY + cellHeight * (row + 0.5);

                if (centerX <= maxX && centerY <= maxY) {
                    anchors.push_back(AnchorPoint{centerX, centerY});
                }
            }
        }

        return anchors;
    }

*/
    // ... Other methods of the Graph class ...
    unordered_set<Node*> getAllNeighbors(int nodeId) {
        unordered_set<Node*> neighbors;
        if (nodes.find(nodeId) != nodes.end()) {
            shared_ptr<Node> node = nodes.at(nodeId);

            for (Node* outNeighbor : node->outNeighbors) {
                neighbors.insert(outNeighbor);
            }
            for (Node* inNeighbor : node->inNeighbors) {
                neighbors.insert(inNeighbor);
            }
        }
        return neighbors;
    }

    const unordered_set<Node*>& getOutNeighbors(int nodeId) {
        static unordered_set<Node*> emptySet; // To handle the case where nodeId is not found

        if (nodes.find(nodeId) != nodes.end()) {
            return nodes.at(nodeId)->outNeighbors;
        } else {
            return emptySet;
        }
    }

    const unordered_set<Node*>& getInNeighbors(int nodeId) {
        static unordered_set<Node*> emptySet; // To handle the case where nodeId is not found

        if (nodes.find(nodeId) != nodes.end()) {
            return nodes.at(nodeId)->inNeighbors;
        } else {
            return emptySet;
        }
    }

    void printGraph() {
        for (const auto& pair : nodes) {
            shared_ptr<Node> node = pair.second;
            if (node) {
                cout << "Node " << node->id << ":\n";
                cout << "  In-Neighbors: ";
                for (const auto& inNeighbor : node->inNeighbors) {
                    cout << inNeighbor->id << " ";
                }
                cout << "\n  Out-Neighbors: ";
                for (const auto& outNeighbor : node->outNeighbors) {
                    cout << outNeighbor->id << " ";
                }
                cout << "\n";
            }
        }
    }


    void printMIPsFromTree(const shared_ptr<TreeNode>& node, vector<int>& currentPath) {
        if (!node) return;

        // Add the current node to the path
        currentPath.push_back(node->graphNode->id);

        if (node->children.empty()) {
            // If it's a leaf node, print the path from this node to the root
            cout << "MIP: ";
            for (auto it = currentPath.rbegin(); it != currentPath.rend(); ++it) {
                cout << *it << " ";
            }
            cout << endl;
        } else {
            // Continue to traverse for each child
            for (const auto& child : node->children) {
                printMIPsFromTree(child, currentPath);
            }
        }

        // Remove the current node before going back up the tree
        currentPath.pop_back();
    }

    void findAllMIPsInMIIA(const shared_ptr<TreeNode>& root) {
        vector<int> currentPath;
        printMIPsFromTree(root, currentPath);
    }

    void printMIPsInMIOA(const shared_ptr<TreeNode>& node, vector<int>& currentPath) {
        if (!node) return;

        // Add the current node to the path
        currentPath.push_back(node->graphNode->id);

        if (node->children.empty()) {
            // If it's a leaf node, print the path from the root to this node
            cout << "MIP: ";
            for (int id : currentPath) {
                cout << id << " ";
            }
            cout << endl;
        } else {
            // Continue to traverse for each child
            for (const auto& child : node->children) {
                printMIPsInMIOA(child, currentPath);
            }
        }

        // Remove the current node before going back up the tree
        currentPath.pop_back();
    }

    void findAllMIPsInMIOA(const shared_ptr<TreeNode>& root) {
        vector<int> currentPath;
        printMIPsInMIOA(root, currentPath);
    }

    double findPropagationProbability(Node* w, Node* u) {
        if (!w) return 0.0;  // Check for null pointer
        for (auto& edge : w->outgoingEdges) {
            if (edge.to == u) {
                return edge.probability;
            }
        }
        return 0.0;  // Return 0 if no edge is found
    }

    double calculateProbabilityOfActivationFromSet(const unordered_set<Node*>& S, Node* w) {
        double probabilityNotActivated = 1.0;

        for (Node* s : S) {
            if (s->outNeighbors.find(w) != s->outNeighbors.end()) {
                // Assuming 'findPropagationProbability' returns the probability of s activating w
                double activationProbability = findPropagationProbability(s, w);
                probabilityNotActivated *= (1 - activationProbability);
            }
        }

        return 1 - probabilityNotActivated;
    }

    //eq. 2 Iq(S,v)
    double calculateInfluenceSpread(const unordered_set<Node*>& S, Node* v, pair<double, double> q) {
        double product = 1.0;

        for (Node* neighbor : v->inNeighbors) {
            double p_wv = findPropagationProbability(neighbor, v); // P(w, v)
            double p_Swv = 0.0; // P(S, w, v) - To be calculated based on the set S and node w

            //  P(S, w, v)
            p_Swv = calculateProbabilityOfActivationFromSet(S, neighbor);

            product *= (1 - p_Swv * p_wv);
        }

        return (1 - product) * f(v, q);
    }

    //eq. 3 Iq(S)
    double calculateTotalInfluence(const unordered_set<Node*>& S, pair<double, double> q) {
        double totalInfluence = 0.0;
        for (auto& pair : nodes) {
            Node* v = pair.second.get();
            totalInfluence += calculateInfluenceSpread(S, v, q); // Assuming this function implements Iq(S, v)
        }
        return totalInfluence;
    }

    //marginal influence Iq(u|S)
    double calculateMarginalInfluence(Node* u, const unordered_set<Node*>& S, pair<double, double> q) {
        if (S.find(u) != S.end()) {
            // If u is already in S, its marginal influence is 0
            return 0.0;
        }

        // Calculate Iq(S)
        double influenceWithoutU = calculateTotalInfluence(S, q);

        // Calculate Iq(S U {u})
        unordered_set<Node*> SWithU = S;
        SWithU.insert(u);
        double influenceWithU = calculateTotalInfluence(SWithU, q);

        // The marginal influence is the difference
        return influenceWithU - influenceWithoutU;
    }   



    //For rule 1!-------------------------------------------------------------------------------------------------------------------------------
    //Anchors

    vector<Node*> getNodesInRange(pair<double, double> anchor, double range) {
        vector<Node*> nodesInRange;

        for (auto& pair : nodes) {
            Node* node = pair.second.get();
            double distance = calculateDistance(node->longitude, node->latitude, anchor.first, anchor.second);

            if (distance <= range) {
                nodesInRange.push_back(node);
            }
        }

        return nodesInRange;
    }

    void precomputeInfluenceAtAnchors(const vector<AnchorPoint>& anchors) {
        for (auto& pair : nodes) {
            Node* node = pair.second.get();
            for (const auto& anchor : anchors) {
                node->influenceAtAnchors[anchor] = calculateTotalInfluence({node}, make_pair(anchor.x, anchor.y));
            }
        }
    }

    double calculateDMax(Node* u, const vector<AnchorPoint>& anchors) {
        double d_max = numeric_limits<double>::max();

        for (const auto& anchor : anchors) {
            double distance = calculateDistance(u->longitude, u->latitude, anchor.x, anchor.y);
            d_max = min(d_max, distance);
        }

        return d_max;
    }

    vector<AnchorPoint> generateAnchorPoints(int numAnchorPoints) {
        vector<AnchorPoint> anchors;

        // Calculate the dimensions of each cell
        int numRows = sqrt(numAnchorPoints);
        int numCols = (numAnchorPoints + numRows - 1) / numRows; // Adjust for rounding

        double cellWidth = (maxX - minX) / numCols;
        double cellHeight = (maxY - minY) / numRows;

        // Generate anchor points at the center of each cell
        for (int row = 0; row < numRows; ++row) {
            for (int col = 0; col < numCols; ++col) {
                double centerX = minX + cellWidth * (col + 0.5);
                double centerY = minY + cellHeight * (row + 0.5);

                if (centerX <= maxX && centerY <= maxY) {
                    anchors.push_back(AnchorPoint{centerX, centerY});
                }
            }
        }

        return anchors;
    }

    AnchorPoint findClosestAnchor(pair<double, double> q, const vector<AnchorPoint>& anchors) {
        double minDistance = numeric_limits<double>::max();
        AnchorPoint closestAnchor = anchors[0];

        for (const auto& anchor : anchors) {
            double distance = calculateDistance(anchor.x, anchor.y, q.first, q.second);
            if (distance < minDistance) {
                minDistance = distance;
                closestAnchor = anchor;
            }
        }

        return closestAnchor;
    }

    pair<double, double> estimateBoundsUsingAnchorPoints(Node* u, const vector<AnchorPoint>& anchors, pair<double, double> q) {
        AnchorPoint closestAnchor = findClosestAnchor(q, anchors);
        double distanceToQuery = calculateDistance(closestAnchor.x, closestAnchor.y, q.first, q.second);

        double influenceAtAnchor = u->influenceAtAnchors[closestAnchor];
        double upperBound = influenceAtAnchor * exp(ALPHA * distanceToQuery);
        double lowerBound = influenceAtAnchor * exp(-ALPHA * distanceToQuery);

        return make_pair(upperBound, lowerBound);
    }


    //grids

    void calculateInfluenceInGrid(Grid& grid) {
        for (auto& pair : nodes) {
            Node* node = pair.second.get();
            if (isNodeInGrid(node, grid)) {
                double influence = calculateTotalInfluence({node}, make_pair((grid.minX + grid.maxX) / 2, (grid.minY + grid.maxY) / 2));
                grid.nodeInfluence[node->id] = influence;
                node->influenceInGrids[grid] = influence;
            }
        }
    }

    bool isNodeInGrid(Node* node, const Grid& grid) {
        return node->longitude >= grid.minX && node->longitude <= grid.maxX &&
               node->latitude >= grid.minY && node->latitude <= grid.maxY;
    }

    void partitionSpaceIntoGrids(int gridRows, int gridCols) {
        double gridWidth = (maxX - minX) / gridCols;
        double gridHeight = (maxY - minY) / gridRows;

        for (int row = 0; row < gridRows; ++row) {
            for (int col = 0; col < gridCols; ++col) {
                Grid grid;
                grid.minX = minX + col * gridWidth;
                grid.maxX = grid.minX + gridWidth;
                grid.minY = minY + row * gridHeight;
                grid.maxY = grid.minY + gridHeight;

                calculateInfluenceInGrid(grid);
            }
        }
    }

    double calculateMaxDistance(const pair<double, double>& q, const Grid& grid) {
        // Corners of the grid
        vector<pair<double, double>> corners = {
            {grid.minX, grid.minY},
            {grid.maxX, grid.minY},
            {grid.minX, grid.maxY},
            {grid.maxX, grid.maxY}
        };

        double maxDist = 0.0;
        for (const auto& corner : corners) {
            double dist = calculateDistance(q.first, q.second, corner.first, corner.second);
            maxDist = max(maxDist, dist);
        }

        return maxDist;
    }

    double calculateMinDistance(const pair<double, double>& q, const Grid& grid) {
        // Check if the query point is inside the grid
        if (q.first >= grid.minX && q.first <= grid.maxX && q.second >= grid.minY && q.second <= grid.maxY) {
            return 0.0; // Minimum distance is zero if the point is inside the grid
        }

        // Check distance to each grid boundary
        double minDist = numeric_limits<double>::max();
        // Left boundary
        minDist = min(minDist, abs(q.first - grid.minX));
        // Right boundary
        minDist = min(minDist, abs(grid.maxX - q.first));
        // Top boundary
        minDist = min(minDist, abs(grid.maxY - q.second));
        // Bottom boundary
        minDist = min(minDist, abs(q.second - grid.minY));

        // Check distance to grid corners
        vector<pair<double, double>> corners = {
            {grid.minX, grid.minY},
            {grid.maxX, grid.minY},
            {grid.minX, grid.maxY},
            {grid.maxX, grid.maxY}
        };

        for (const auto& corner : corners) {
            double dist = calculateDistance(q.first, q.second, corner.first, corner.second);
            minDist = min(minDist, dist);
        }

        return minDist;
    }

    pair<double, double> estimateBoundsUsingRegion(Node* u, pair<double, double> q) {
        double upperBound = 0.0, lowerBound = 0.0;
        for (const auto& pair : u->influenceInGrids) {
            const Grid& grid = pair.first;
            double influence = pair.second;
            
            double dmax = calculateMaxDistance(q, grid);
            double dmin = calculateMinDistance(q, grid);

            upperBound += influence * DECAY_PARAMETER * exp(-ALPHA * dmin);
            lowerBound += influence * DECAY_PARAMETER * exp(-ALPHA * dmax);
        }
        return make_pair(upperBound, lowerBound);
    }


    //Fused
    
    // Method to select significant nodes
    void selectSignificantNodes(unordered_set<Node*>& significantNodes, unordered_set<Node*>& lessInfluentialNodes) {
        vector<pair<double, Node*>> nodeInfluences;

        // Calculate total influence for each node
        for (auto& pair : nodes) {
            Node* node = pair.second.get();
            double influence = calculateTotalInfluence({node}, make_pair(0, 0)); // Assuming query location is (0, 0)
            nodeInfluences.emplace_back(influence, node);
        }

        // Sort nodes by their influence
        sort(nodeInfluences.begin(), nodeInfluences.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Select top-TAU nodes as significant
        for (int i = 0; i < min(TAU, static_cast<int>(nodeInfluences.size())); ++i) {
            significantNodes.insert(nodeInfluences[i].second);
        }

        // Remaining nodes are less influential
        for (int i = TAU; i < static_cast<int>(nodeInfluences.size()); ++i) {
            lessInfluentialNodes.insert(nodeInfluences[i].second);
        }
    }

    // Fused approach to estimate bounds
    pair<double, double> estimateBoundsFused(Node* u, pair<double, double> q, const vector<AnchorPoint>& anchors) {
        auto boundsAnchor = estimateBoundsUsingAnchorPoints(u, anchors, q);
        auto boundsRegion = estimateBoundsUsingRegion(u, q);

        double upperBound = min(boundsAnchor.first, boundsRegion.first);
        double lowerBound = max(boundsAnchor.second, boundsRegion.second);

        return make_pair(upperBound, lowerBound);
    }

    //Rule 2--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    double calculateMarginalInfluenceUpperBound(int nodeU, const unordered_set<int>& seedSet, Graph& graph, double delta, double theta, pair<double, double> q) {
        double marginalInfluenceUpperBound = 0.0;
        unordered_map<int, double> partitionInfluence;

        // BFS to traverse MIOA tree
        queue<shared_ptr<TreeNode>> queue;
        shared_ptr<TreeNode> mioaRootU = graph.calculateMIOA(nodeU, theta, q);
        if (mioaRootU) {
            queue.push(mioaRootU);
        }

        while (!queue.empty()) {
            shared_ptr<TreeNode> currentNode = queue.front();
            queue.pop();

            // Calculate influence of nodeU on the current node
            unordered_set<Node*> singletonSet = {currentNode->graphNode.get()};
            double influence = graph.calculateTotalInfluence(singletonSet, make_pair(currentNode->graphNode->longitude, currentNode->graphNode->latitude));

            int partitionIndex = static_cast<int>(log(influence / delta) / log(theta));
            partitionInfluence[partitionIndex] += influence;

            for (auto& child : currentNode->children) {
                queue.push(child);
            }
        }

        double seedInfluence = 0.0;
        for (int seed : seedSet) {
            if (graph.nodes.find(seed) != graph.nodes.end()) {
                Node* seedNode = graph.nodes.at(seed).get();
                unordered_set<Node*> singletonSeedSet = {seedNode};
                seedInfluence += graph.calculateTotalInfluence(singletonSeedSet, make_pair(seedNode->longitude, seedNode->latitude));
            }
        }
        seedInfluence = min(seedInfluence, 1.0);

        // Aggregate influence values for each partition
        for (const auto& partition : partitionInfluence) {
            marginalInfluenceUpperBound += (1 - seedInfluence) * partition.second;
        }

        return marginalInfluenceUpperBound;
    }


    // Rule 3 ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    vector<Viewpoint> generateViewpoints(int numViewpoints) {
        vector<Viewpoint> viewpoints;

        // Calculate the dimensions of each cell
        int numRows = sqrt(numViewpoints);
        int numCols = (numViewpoints + numRows - 1) / numRows; // Adjust for rounding

        double cellWidth = (maxX - minX) / numCols;
        double cellHeight = (maxY - minY) / numRows;

        // Generate viewpoints at the center of each cell
        for (int row = 0; row < numRows; ++row) {
            for (int col = 0; col < numCols; ++col) {
                double centerX = minX + cellWidth * (col + 0.5);
                double centerY = minY + cellHeight * (row + 0.5);

                if (centerX <= maxX && centerY <= maxY) {
                    viewpoints.emplace_back(centerX, centerY);
                }
            }
        }

        return viewpoints;
    }

    //helper function for to set conversion
    unordered_set<Node*> convertVectorToUnorderedSet(const vector<Node*>& vec) {
        unordered_set<Node*> uset(vec.begin(), vec.end());
        return uset;
    }

    bool shouldTerminateEarly(Node* u, const unordered_set<Node*>& Si_c, const vector<Node*>& Svi, double beta, Graph& graph, const pair<double, double>& q) {
        double marginalInfluenceU = graph.calculateMarginalInfluence(u, Si_c, q);
        unordered_set<Node*> Svi_set = convertVectorToUnorderedSet(Svi);
        double influenceSvi = graph.calculateTotalInfluence(Svi_set, q); 
        double influenceSi_1 = graph.calculateTotalInfluence(Si_c, q);

        return marginalInfluenceU >= influenceSvi / beta - influenceSi_1;
    }
   
    bool shouldPruneNode(Node* u, const unordered_set<Node*>& Si_c, const vector<Node*>& Svi, int i, Graph& graph, const pair<double, double>& q) {
        if (Si_c.find(u) != Si_c.end()) {
            return true; // Node u is already in Si_1
        }

        double marginalInfluenceU = graph.calculateMarginalInfluence(u, Si_c, q);
        double marginalInfluenceSvi = (i < Svi.size()) ? graph.calculateMarginalInfluence(Svi[i], Si_c, q) : 0;

        return marginalInfluenceU <= marginalInfluenceSvi;
    }
  

    //Old MIA stuff ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/*     void calculateAPIteratively(shared_ptr<TreeNode> root, const unordered_set<int>& S, double theta, unordered_map<int, double>& alphaCache) {
        if (!root) return;

        // Initialize all node activation probabilities to zero
        for (auto& pair : nodes) {
            alphaCache[pair.first] = (S.find(pair.first) != S.end()) ? 1.0 : 0.0;
        }

        // Post-order traversal stack
        stack<shared_ptr<TreeNode>> postOrderStack;
        stack<shared_ptr<TreeNode>> stack;
        stack.push(root);
        unordered_set<shared_ptr<TreeNode>> visited;

        while (!stack.empty()) {
            shared_ptr<TreeNode> node = stack.top();
            stack.pop();
            postOrderStack.push(node);
            
            for (auto& child : node->children) {
                if (visited.find(child) == visited.end()) {
                    stack.push(child);
                    visited.insert(child);
                }
            }
        }

        // Calculate activation probabilities in post-order
        while (!postOrderStack.empty()) {
            shared_ptr<TreeNode> treeNode = postOrderStack.top();
            postOrderStack.pop();
            shared_ptr<Node> graphNode = treeNode->graphNode;

            if (S.find(graphNode->id) != S.end()) continue; // Skip seed nodes

            double product = 1.0;
            for (auto& neighbor : graphNode->inNeighbors) {
                double pp_wu = findPropagationProbability(neighbor, graphNode.get());
                double ap_w = alphaCache[neighbor->id];
                product *= (1 - ap_w * pp_wu);
            }

            alphaCache[graphNode->id] = 1 - product;
        }
    }

    shared_ptr<Node> getNode(int nodeId) {
        if (nodes.find(nodeId) != nodes.end()) {
            return nodes[nodeId];
        } else {
            return nullptr;  // Return nullptr if node does not exist
        }
    }


    void calculateAlphaIterativelyV3(shared_ptr<TreeNode> root, const unordered_set<int>& S, double theta, unordered_map<int, double>& alphaCache) {
        if (!root) return;

        // Initialize all node influence values to zero, except for seed nodes
        for (auto& pair : nodes) {
            alphaCache[pair.first] = (S.find(pair.first) != S.end()) ? 1.0 : 0.0;
        }

        // BFS to traverse MIIA tree from root to leaves
        queue<shared_ptr<TreeNode>> q;
        q.push(root);

        while (!q.empty()) {
            shared_ptr<TreeNode> treeNode = q.front();
            q.pop();
            shared_ptr<Node> graphNode = treeNode->graphNode;

            // Skip seed nodes
            if (S.find(graphNode->id) != S.end()) continue;

            double alpha = 0.0;
            if (!graphNode->inNeighbors.empty()) {
                // Compute influence value for this node
                for (auto& inNeighbor : graphNode->inNeighbors) {
                    double pp_wu = findPropagationProbability(inNeighbor, graphNode.get());
                    alpha += alphaCache[inNeighbor->id] * pp_wu;
                }
            }

            // Update influence value for the node
            alphaCache[graphNode->id] = alpha;

            // Add children to queue for BFS
            for (auto& child : treeNode->children) {
                q.push(child);
            }
        }
    } */
/* 
    unordered_set<int> MIA(int k, double theta) {
        unordered_set<int> S;
        unordered_map<int, double> IncInf;
        unordered_map<int, double> apCache;
        unordered_map<int, double> alphaCache;

        // Initialization
        for (auto& pair : nodes) {
            int v = pair.first;
            IncInf[v] = 0;

            auto miiaRoot = calculateMIIA(v, theta);
            auto mioaRoot = calculateMIOA(v, theta);

            calculateAPIteratively(miiaRoot, S, theta, apCache); // Alg. 2
            calculateAlphaIterativelyV3(miiaRoot, S, theta, alphaCache); // Alg. 3

            for (auto& nodePair : nodes) {
                int u = nodePair.first;
                if (miiaRoot->graphNode->id == u) {
                    IncInf[u] += alphaCache[v] * (1 - apCache[u]);
                }
            }
        }

        // Main loop
        for (int i = 0; i < k; ++i) {
            // Find u with maximum IncInf
            int u_max = -1;
            double maxInf = -1;
            for (auto& pair : IncInf) {
                if (S.find(pair.first) == S.end() && pair.second > maxInf) {
                    maxInf = pair.second;
                    u_max = pair.first;
                }
            }

            // Update incremental influence spreads
            auto mioaRootU = calculateMIOA(u_max, theta);
            for (auto& v : mioaRootU->children) { // For each v in MIOA(u, θ) \ S
                auto miiaRootV = calculateMIIA(v->graphNode->id, theta);
                for (auto& w : miiaRootV->children) { // For each w in MIIA(v, θ) \ S
                    IncInf[w->graphNode->id] -= alphaCache[v->graphNode->id] * (1 - apCache[w->graphNode->id]);
                }
            }

            S.insert(u_max);

            // Recompute ap and alpha for affected nodes
            for (auto& v : mioaRootU->children) { // For each v in MIOA(u, θ) \ S
                auto miiaRootV = calculateMIIA(v->graphNode->id, theta);
                calculateAPIteratively(miiaRootV, S, theta, apCache); // Alg. 2
                calculateAlphaIterativelyV3(miiaRootV, S, theta, alphaCache); // Alg. 3

                for (auto& w : miiaRootV->children) { // For each w in MIIA(v, θ) \ S
                    IncInf[w->graphNode->id] += alphaCache[v->graphNode->id] * (1 - apCache[w->graphNode->id]);
                }
            }
        }

        return S;
    } */

};

int main() {
    Graph graph;

    // Create a larger graph:
    // Example connections (directed edges):
    // 1 -> 2, 3
    // 2 -> 4, 5
    // 3 -> 6
    // 4 -> 7
    // 5 -> 7, 8
    // 6 -> 7
    graph.addEdge(1, 2, 0, 0);
    graph.addEdge(1, 3, 0, 0);
    graph.addEdge(2, 4, 0, 0);
    graph.addEdge(2, 5, 0, 0);
    graph.addEdge(3, 6, 0, 0);
    graph.addEdge(4, 7, 0, 0);
    graph.addEdge(5, 7, 0, 0);
    graph.addEdge(5, 8, 0, 0);
    graph.addEdge(6, 7, 0, 0);
    graph.addEdge(1, 7, 0, 0);
    graph.addEdge(7, 3, 0, 0);
    graph.addEdge(3, 1, 0, 0);
    graph.addEdge(7, 5, 0, 0);
    graph.addEdge(5, 2, 0, 0);
    graph.addEdge(2, 1, 0, 0);

    // Test printing the graph
    cout << "Graph structure:\n";
    graph.printGraph();

    /*
    // Testing CalculateMIP from Node 1 to Node 7
    Node* start = graph.getNodeById(7);
    Node* end = graph.getNodeById(1);
    cout<< "Main start and end " << start << ' ' << end;
    //MIP* mip = graph.CalculateMIP(start, end);
    auto mipAndPredecessors = graph.CalculateMIP(start,end);
    MIP* mip = mipAndPredecessors.first;
    auto& Predecessor = mipAndPredecessors.second;
    cout << "\nMaximum Influence Path (MIP) from Node 1 to Node 7:\n";
    for (Node* node : mip->path) {
        cout << node->id << " ";
    }
    cout << "\nPropagation Probability: " << mip->propagationProbability << "\n";
    delete mip; // Clean up the dynamically allocated MIP

    // Test calculateMIIA
    double theta = 0.01; // Example threshold
    cout << "\nMIIA Tree (DFS Traversal) for Node 7:\n";
    shared_ptr<TreeNode> miiaRoot = graph.calculateMIIA(7, theta);
    if (miiaRoot) {
        miiaRoot->DFS();
        cout << "Children" ;
    } else {
        cout << "MIIA Tree not found for Node 7.\n";
    }

    // Test calculateMIOA
    cout << "\nMIOA Tree (DFS Traversal) for Node 1:\n";
    shared_ptr<TreeNode> mioaRoot = graph.calculateMIOA(1, theta);
    if (mioaRoot) {
        mioaRoot->DFS();
    } else {
        cout << "MIOA Tree not found for Node 1.\n";
    }

    cout<<"MIIA :\n";
    shared_ptr<TreeNode> miiaRoot2 = graph.calculateMIIA(7, theta);
    graph.findAllMIPsInMIIA(miiaRoot2);

    cout<<"MIOA :\n";
    shared_ptr<TreeNode> mioaRoot2= graph.calculateMIOA(1, theta);
    graph.findAllMIPsInMIOA(mioaRoot2);

    

     // Define the seed set
    unordered_set<int> S = {1, 3};

    // Set theta threshold
    double theta = 0.01;

    // Initialize a cache for storing activation probabilities
    unordered_map<int, double> alphaCache;

    try {
        // Calculate and print activation probabilities for all nodes
        cout << "Activation probabilities:" << endl;
        for (const auto& pair : graph.nodes) {
            shared_ptr<Node> node = pair.second;
            if (node) {
                double alpha = graph.calculateAlpha(node, S, theta, alphaCache);
                cout << "Node " << node->id << ": " << alpha << endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }
    
   
    // Define the seed set
    unordered_set<int> S = {1, 3};

    // Set theta threshold
    double theta = 0.01;

    // Initialize a cache for storing activation probabilities
    unordered_map<int, double> alphaCache;

    // Calculate MIIA for a target node
    shared_ptr<TreeNode> miiaRoot = graph.calculateMIIA(7, theta);

    // Calculate and print activation probabilities using the iterative method
    graph.calculateAPIteratively(miiaRoot, S, theta, alphaCache);
    cout << "Activation probabilities (Iterative):" << endl;
    for (const auto& pair : graph.nodes) {
        shared_ptr<Node> node = pair.second;
        if (node) {
            double alpha = alphaCache[node->id];
            cout << "Node " << node->id << ": " << alpha << endl;
        }
    }

    unordered_map<int, double> alphaCache2;

    // Specify the nodes u and v for which you want to calculate alpha(u, v)
    int u = 2; // Example node u
    int v = 6; // Example node v

    // Calculate MIIA for the target node v
    shared_ptr<TreeNode> miiaRoot_2 = graph.calculateMIIA(v, theta);

    // Calculate influence values using the iterative method
    graph.calculateAlphaIterativelyV3(miiaRoot_2, S, theta, alphaCache2);

    // Print the specific influence value alpha(u, v)
    cout << "Influence value alpha(" << u << ", " << v << "): " 
         << alphaCache2[u] << endl;  

    */

    int k = 3; // Number of seeds to find
    double theta = 0.01; // Threshold

    /* unordered_set<int> seedSet = graph.MIA(k, theta);

    cout << "Seed set: ";
    for (int node : seedSet) {
        cout << node << " ";
    }
    cout << endl; */

    return 0;
}