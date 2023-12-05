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
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;

struct TreeNode;
struct Node;
struct Edge;
struct CELFNode;

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
    vector<string> checkInTime;
    vector<pair<double, double>> locations;
    // For rule 1
    unordered_map<AnchorPoint, double> influenceAtAnchors;
    unordered_map<Grid, double> influenceInGrids;

    // Default constructor
    Node() : id(-1), longitude(0.0), latitude(0.0) {}

    // Parameterized constructor
    Node(int id, double longi = 0.0, double lati = 0.0) : id(id), longitude(longi), latitude(lati) {}

    void addCheckIn(const string& time, double lat, double lon) {
        checkInTime.push_back(time);
        locations.push_back(make_pair(lat, lon));
    }

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
        std::cout << "Node ID: " << graphNode->id << endl;
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

/* bool operator<(const CELFNode& a, const CELFNode& b) {
    return a.marginalGain < b.marginalGain;
}
 */
struct CELFNode {
    Node* node;
    double marginalGain;
    int round;

    CELFNode(Node* node, double marginalGain, int round) 
        : node(node), marginalGain(marginalGain), round(round) {}
};

class Graph {
public:
    unordered_map<int, shared_ptr<Node>> nodes;
    double minX = -200, maxX = 200, minY = -200, maxY = 200;

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
                std::cout << "Node " << node->id << ":\n";
                std::cout << "  In-Neighbors: ";
                for (const auto& inNeighbor : node->inNeighbors) {
                    std::cout << inNeighbor->id << " ";
                }
                std::cout << "\n  Out-Neighbors: ";
                for (const auto& outNeighbor : node->outNeighbors) {
                    std::cout << outNeighbor->id << " ";
                }
                std::cout << "\n";
            }
        }
    }


    void printMIPsFromTree(const shared_ptr<TreeNode>& node, vector<int>& currentPath) {
        if (!node) return;

        // Add the current node to the path
        currentPath.push_back(node->graphNode->id);

        if (node->children.empty()) {
            // If it's a leaf node, print the path from this node to the root
            std::cout << "MIP: ";
            for (auto it = currentPath.rbegin(); it != currentPath.rend(); ++it) {
                std::cout << *it << " ";
            }
            std::cout << endl;
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
            std::cout << "MIP: ";
            for (int id : currentPath) {
                std::cout << id << " ";
            }
            std::cout << endl;
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

    //helper function for set conversion
    unordered_set<Node*> convertVectorToUnorderedSet(const vector<Node*>& vec) {
        unordered_set<Node*> uset(vec.begin(), vec.end());
        return uset;
    }

    /*bool shouldTerminateEarly(Node* u, const unordered_set<Node*>& Si_c, const vector<Node*>& Svi, double beta, Graph& graph, const pair<double, double>& q) {
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
    }*/

bool shouldTerminateEarly(Node* u, const unordered_set<Node*>& Si_c, const vector<Node*>& Svi, double beta, Graph& graph, const pair<double, double>& q) {
    double marginalInfluenceU = graph.calculateMarginalInfluence(u, Si_c, q);
    double totalInfluenceSi_c = graph.calculateTotalInfluence(Si_c, q);
    
    // Terminate early if the marginal influence of u is not sufficiently larger than the current total influence
    return marginalInfluenceU <= beta * totalInfluenceSi_c;
}

bool shouldPruneNode(Node* u, const unordered_set<Node*>& Si_c, const vector<Node*>& Svi, int i, Graph& graph, const pair<double, double>& q) {
    if (Si_c.find(u) != Si_c.end()) {
        return true; // Node u is already in Si_c
    }

    double marginalInfluenceU = graph.calculateMarginalInfluence(u, Si_c, q);

    if (i < Svi.size()) {
        double marginalInfluenceSvi = graph.calculateMarginalInfluence(Svi[i], Si_c, q);
        return marginalInfluenceU <= marginalInfluenceSvi;
    }

    return false; // Do not prune if there are no more candidates in Svi
}

};

/*void setWCModel(Graph& graph) {
    for (auto& nodePair : graph.nodes) {
        auto& node = nodePair.second;
        int incomingNeighborsCount = node->inNeighbors.size();
        if (incomingNeighborsCount > 0) {
            double probability = 1.0 / incomingNeighborsCount;
            for (auto& edge : node->incomingEdges) {
                edge.probability = probability;
            }
        }
    }
}*/

void setWCModel(Graph& graph) {
    // Iterate through each node in the graph
    for (auto& nodePair : graph.nodes) {
        auto& node = nodePair.second;

        // Calculate in-degree for the current node
        int inDegree = node->inNeighbors.size();

        // If the node has incoming edges
        if (inDegree > 0) {
            // Calculate the probability for the incoming edges
            double probability = 1.0 / static_cast<double>(inDegree);

            // Set the probability for each incoming edge
            for (Edge& edge : node->incomingEdges) {
                edge.probability = probability;
            }
        }
    }
}

void setTCModel(Graph& graph) {
    const vector<double> probabilities = {0.1, 0.01, 0.001};
    srand(time(nullptr));  // Seed for random number generator

    for (auto& nodePair : graph.nodes) {
        auto& node = nodePair.second;
        for (auto& edge : node->outgoingEdges) {
            int randomIndex = rand() % probabilities.size();  // Random index 0, 1, or 2
            edge.probability = probabilities[randomIndex];
        }
    }
}

//PMIA
//helper functions for algorithm 3
/*double findPropagationProbability(Node* from, Node* to) {
    for (auto& edge : from->outgoingEdges) {
        if (edge.to == to) {
            return edge.probability;
        }
    }
    return 0.0;  // Default probability if no edge is found
}

double calculateAP(Node* u, const unordered_set<Node*>& S, const unordered_map<Node*, shared_ptr<TreeNode>>& miiaTrees) {
    if (S.find(u) != S.end()) {
        return 1.0;  // u is a seed itself
    }

    double ap = 0.0;
    auto parentNode = miiaTrees.at(u)->parent.lock();
    if (parentNode) {
        Node* parent = parentNode->graphNode.get();
        double pp_parent_u = findPropagationProbability(parent, u);
        double product = 1.0;
        for (Node* inNeighbor : parent->inNeighbors) {
            if (inNeighbor != u && S.find(inNeighbor) == S.end()) {
                double pp_inNeighbor_parent = findPropagationProbability(inNeighbor, parent);
                product *= (1 - calculateAP(inNeighbor, S, miiaTrees) * pp_inNeighbor_parent);
            }
        }
        ap = calculateAP(parent, S, miiaTrees) * pp_parent_u * product;
    }

    return ap;
}

//algorithm 3 in paper ref 5, which is used in PMIA algorithm
double calculateAlpha(Node* v, Node* u, const unordered_map<Node*, shared_ptr<TreeNode>>& miiaTrees, const unordered_set<Node*>& S) {
    if (u == v) {
        return 1.0;
    }

    double alpha = 0.0;
    for (auto& childNode : miiaTrees.at(u)->children) {
        Node* w = childNode->graphNode.get();
        if (S.find(w) != S.end()) {
            // Influence of u to v is blocked by seed w
            return 0.0;
        } else {
            double pp_uw = findPropagationProbability(u, w); // Probability of u influencing w
            double product = 1.0;
            for (Node* u0 : w->inNeighbors) {
                if (u0 != u) {
                    double ap_u0 = calculateAP(u0, S, miiaTrees); // This function needs to be defined
                    double pp_u0w = findPropagationProbability(u0, w);
                    product *= (1 - ap_u0 * pp_u0w);
                }
            }
            alpha += calculateAlpha(v, w, miiaTrees, S) * pp_uw * product;
        }
    }
    return alpha;
}

//PMIA main implementation (see this in reference 5 to check each step)
unordered_set<Node*> PMIA(Graph& graph, int k, double theta) {
    unordered_set<Node*> S;
    unordered_map<Node*, double> IncInf; // Incremental Influence for each node
    unordered_map<Node*, shared_ptr<TreeNode>> miiaTrees, mioaTrees;

    // Initialize Incremental Influence to 0
    for (auto& nodePair : graph.nodes) {
        IncInf[nodePair.second.get()] = 0;
    }

    // Step 4-11: Initialization
    for (auto& nodePair : graph.nodes) {
        Node* v = nodePair.second.get();
        miiaTrees[v] = graph.calculateMIIA(v->id, theta, {0, 0}); // Assuming default query location

        for (auto& childNode : miiaTrees[v]->getAllNodes()) {
            Node* u = childNode.get();
            double alpha = calculateAlpha(v, u, miiaTrees, S); // Updated to use the new function
            IncInf[u] += alpha * (1 - calculateAP(u, S, miiaTrees)); // Function to calculate ap(w, S, PMIIA(v, θ, S))
        }
    }

    // Step 13-34: Main loop
    for (int i = 0; i < k; ++i) {
        // Select the node with the highest incremental influence
        Node* maxNode = nullptr;
        double maxIncInf = -1;
        for (auto& incInfPair : IncInf) {
            if (S.find(incInfPair.first) == S.end() && incInfPair.second > maxIncInf) {
                maxIncInf = incInfPair.second;
                maxNode = incInfPair.first;
            }
        }

        if (maxNode == nullptr) {
            break;
        }

        S.insert(maxNode);
        IncInf[maxNode] = 0; // Resetting its own influence as it's now selected

        // Update Incremental Influences
        mioaTrees[maxNode] = graph.calculateMIOA(maxNode->id, theta, {0, 0});
        for (auto& targetNode : mioaTrees[maxNode]->getAllNodes()) {
            Node* v = targetNode.get();
            
            for (auto& u : graph.getAllNeighbors(v->id)) {
                if (S.find(u) == S.end()) {
                    IncInf[u] -= calculateAlpha(v, u, miiaTrees, S) * (1 - calculateAP(u, S, miiaTrees));
                }
            }

            if (v != maxNode) {
                miiaTrees[v] = graph.calculateMIIA(v->id, theta, {0, 0});
                for (auto& u : graph.getAllNeighbors(v->id)) {
                    if (S.find(u) == S.end()) {
                        IncInf[u] += calculateAlpha(v, u, miiaTrees, S) * (1 - calculateAP(u, S, miiaTrees));
                    }
                }
            }
        }
    }
    return S;
}*/

//CELF++ implementation
/* 
unordered_set<Node*> CELFPlusPlus(Graph& graph, int k) {
    priority_queue<CELFNode> pq;
    unordered_set<Node*> seedSet;
    unordered_map<Node*, int> lastUpdate;

    // Initialize with marginal gain = total influence
    for (auto& nodePair : graph.nodes) {
        Node* node = nodePair.second.get();
        double influence = graph.calculateTotalInfluence({node}, {0, 0}); // Assuming default query location
        pq.emplace(node, influence, 0);
        lastUpdate[node] = 0;
    }

    // Greedily find k nodes
    int currentRound = 0;
    while (seedSet.size() < k) {
        CELFNode topNode = pq.top();
        pq.pop();

        if (lastUpdate[topNode.node] == currentRound) {
            // This node's marginal gain is up-to-date
            seedSet.insert(topNode.node);
            currentRound++;
        } else {
            // Recalculate the marginal gain
            unordered_set<Node*> tempSeedSet = seedSet;
            tempSeedSet.insert(topNode.node);
            double newInfluence = graph.calculateTotalInfluence(tempSeedSet, {0, 0});
            double originalInfluence = graph.calculateTotalInfluence(seedSet, {0, 0});
            double marginalGain = newInfluence - originalInfluence;

            pq.emplace(topNode.node, marginalGain, currentRound);
            lastUpdate[topNode.node] = currentRound;
        }
    }
    return seedSet;
} */

// function to calculate algorithm running time
double measureRunningTime(function<void()> func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    return duration.count(); // Running time in seconds
}

// influence spread measurement
double measureInfluenceSpread(Graph& graph, const unordered_set<Node*>& seedNodes) {
    unordered_set<int> influencedNodes; // To keep track of influenced nodes
    queue<Node*> nodesToProcess; // Queue for Breadth-First Search (BFS)

    // Initialize: add all seed nodes to the queue and mark them as influenced
    for (Node* seedNode : seedNodes) {
        nodesToProcess.push(seedNode);
        influencedNodes.insert(seedNode->id);
    }

    while (!nodesToProcess.empty()) {
        Node* currentNode = nodesToProcess.front();
        nodesToProcess.pop();

        // Traverse all outgoing edges of the current node
        for (Edge& edge : currentNode->outgoingEdges) {
            Node* neighbor = edge.to;

            // Check if the neighbor is already influenced
            if (influencedNodes.find(neighbor->id) == influencedNodes.end()) {
                double randomValue = static_cast<double>(rand()) / RAND_MAX;

                // Influence the neighbor based on the edge's probability
                if (randomValue < edge.probability) {
                    nodesToProcess.push(neighbor);
                    influencedNodes.insert(neighbor->id);
                }
            }
        }
    }

    // The influence spread is the number of influenced nodes
    return static_cast<double>(influencedNodes.size());
}


int main() {
    // Seed the random number generator, for TC/WC model
    srand(static_cast<unsigned int>(time(nullptr)));

    // Check if text file is provided in the command line
    // note load ckeckin and locations file first, then the neighbor file
    /* if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <network_file> <additional_data_file>\n";
        return 1;


    } */

    try {
        //string networkFileName = argv[1];
        //string additionalDataFileName = argv[2];
        string networkFileName = "social_graph_edges2.txt";
        string additionalDataFileName = "social_graph_nodes.txt";
        Graph networkGraph;

        string line;
        istringstream lineStream;

        ifstream additionalDataFile(additionalDataFileName);
        if (!additionalDataFile) {
            throw runtime_error("Error: Could not open additional data file " + additionalDataFileName);
        }

        while(getline(additionalDataFile, line)){
            lineStream.clear();
            lineStream.str(line);
            int id;
            double lat, lon;
            string timestamp, locationId;
            if (lineStream >> id >> timestamp >> lat >> lon >> locationId){
                // If the node does not exist, create it
                if (networkGraph.nodes.find(id) == networkGraph.nodes.end()) {
                    networkGraph.nodes[id] = make_shared<Node>(id);
                }
                networkGraph.nodes[id]->addCheckIn(timestamp, lat, lon);
            }
        }
        additionalDataFile.close();

        ifstream networkFile(networkFileName);
        if (!networkFile) {
            throw runtime_error("Error: Could not open network file " + networkFileName);
        }

        while(getline(networkFile, line)){
            lineStream.clear();
            lineStream.str(line);
            int id, neighborId;

            if(lineStream >> id >> neighborId){
                // Ensure both nodes exist
                if (networkGraph.nodes.find(id) == networkGraph.nodes.end()) {
                    networkGraph.nodes[id] = make_shared<Node>(id);
                }
                if (networkGraph.nodes.find(neighborId) == networkGraph.nodes.end()) {
                    networkGraph.nodes[neighborId] = make_shared<Node>(neighborId);
                }
                // Use Graph's addEdge function to handle neighbor relationships
                networkGraph.addEdge(id, neighborId, 0.0, 0.0); // Assuming default lat/long are fine here
            }
        }
        networkFile.close();
        
        int seedSetSize = 50; // Varies from 10 to 50
        int numAnchorPoints = 200; // Varies from 100 to 300
        int numViewPoints = 1000; // Varies from 500 to 1500
        double theta = 0.001; // Threshold for influence
        double delta = 0.01; // Parameter for partitioning influences
        double beta = 0.5; // Parameter for early termination condition

        int k = 10; //number of seeds select, can be varied

        // Set propagation probability model (WC or TC), choose one model a time during experiment
        setWCModel(networkGraph);
        cout << "Using WC model: " << endl;
        //setTCModel(networkGraph); // <-remove comment if elment this, and comment the other
        //cout << "Using TC model: " << endl;

        // Run experiments for each algorithm
        // Effectiveness evaluation
        // Compare the algorithms based on influence spread, response time

        // Measure running time
        /*unordered_set<Node*> selectedSeedsPMIA;
        double runningTimePMIA = measureRunningTime([&] {
            // Run PMIA
            selectedSeedsPMIA = PMIA(networkGraph, k, theta);
            cout << "Here2\n";
            std::cout << "Selected Seeds: ";
            for (const auto& seed : selectedSeedsPMIA) {
                cout << "Here2\n";
                std::cout << seed->id << " ";
            }
            std::cout << endl;
        });
        cout << "Here4\n";*/

        /* unordered_set<Node*> selectedSeedsCELFPP;
        double runningTimeCELFPP = measureRunningTime([&] {
            // Run CELF++
            selectedSeedsCELFPP = CELFPlusPlus(networkGraph, k);
            std::cout << "Selected Seeds CELF++: ";
            for (const auto& seed : selectedSeedsCELFPP) {
                std::cout << seed->id << " ";
            }
            std::cout << endl;
        }); */
        
        unordered_set<Node*> selectedSeedsPRI;
        double runningTimePRI = measureRunningTime([&] {
            // Run PRI
            vector<AnchorPoint> anchors = networkGraph.generateAnchorPoints(numAnchorPoints);
            networkGraph.precomputeInfluenceAtAnchors(anchors);

            // Run PRI using Rule 1
            unordered_set<Node*> significantNodes, lessInfluentialNodes;
            networkGraph.selectSignificantNodes(significantNodes, lessInfluentialNodes);

            std::cout << "Significant Nodes (PRI Rule 1): ";
            for (const auto& node : significantNodes) {
                if (node) std::cout << node->id << " ";
            }
            std::cout << endl;

            // Select top-k nodes from significantNodes as seeds
            vector<Node*> seedCandidates(significantNodes.begin(), significantNodes.end());
            sort(seedCandidates.begin(), seedCandidates.end(), [&networkGraph, &anchors](Node* a, Node* b) {
                return networkGraph.estimateBoundsFused(a, {0, 0}, anchors).first > 
                    networkGraph.estimateBoundsFused(b, {0, 0}, anchors).first;
            });

            for (int i = 0; i < min(k, static_cast<int>(seedCandidates.size())); ++i) {
                selectedSeedsPRI.insert(seedCandidates[i]);
            }

            std::cout << "Selected Seeds (PRI Rule 1): ";
            for (const auto& seed : selectedSeedsPRI) {
                std::cout << seed->id << " ";
            }
            std::cout << endl;
        });

        unordered_set<Node*> selectedSeedsPRII;
        double runningTimePRII = measureRunningTime([&] {
            // Run PRII
            unordered_set<int> seedIds; // To store IDs of selected seeds

            for (int i = 0; i < k; ++i) {
                double maxUpperBound = -1.0;
                Node* candidate = nullptr;

                for (auto& nodePair : networkGraph.nodes) {
                    Node* node = nodePair.second.get();
                    if (seedIds.find(node->id) == seedIds.end()) {
                        double upperBound = networkGraph.calculateMarginalInfluenceUpperBound(node->id, seedIds, networkGraph, delta, theta, {0, 0});
                        if (upperBound > maxUpperBound) {
                            maxUpperBound = upperBound;
                            candidate = node;
                        }
                    }
                }

                if (candidate) {
                    selectedSeedsPRII.insert(candidate);
                    seedIds.insert(candidate->id);
                }
            }

            std::cout << "Selected Seeds (PRII Rule 2): ";
            for (const auto& seed : selectedSeedsPRII) {
                std::cout << seed->id << " ";
            }
            std::cout << endl;
        });

        unordered_set<Node*> selectedSeedsPRIII;
        double runningTimePRIII = measureRunningTime([&] {
            // Run PRIII
            // Generate viewpoints
            vector<Viewpoint> viewpoints = networkGraph.generateViewpoints(1000); // Adjust the number of viewpoints as needed

            // Run PRIII using Rule 3
            // Reset selectedSeeds and seedIds for a new selection
            unordered_set<int> seedIds;
            vector<Node*> Svi; // Vector to store potential seeds for evaluation

            for (int i = 0; i < k; ++i) {
                Node* bestNode = nullptr;
                double bestInfluence = -1.0;
                double maxUpperBound = -1.0;
                Node* candidate = nullptr;
                for (auto& nodePair : networkGraph.nodes) {
                    Node* node = nodePair.second.get();
                    if (seedIds.find(node->id) == seedIds.end() && !networkGraph.shouldPruneNode(node, selectedSeedsPRIII, Svi, i, networkGraph, {0, 0})) {
                        double influence = networkGraph.calculateMarginalInfluence(node, selectedSeedsPRIII, {0, 0});
                        if (influence > bestInfluence && !networkGraph.shouldTerminateEarly(node, selectedSeedsPRIII, Svi, beta, networkGraph, {0, 0})) {
                            bestInfluence = influence;
                            bestNode = node;
                        }
                    }
                }

                if (bestNode) {
                    selectedSeedsPRIII.insert(bestNode);
                    seedIds.insert(bestNode->id);
                    Svi.push_back(bestNode); // Add the selected node to Svi for future pruning checks
                }
            }

            std::cout << "Selected Seeds (PRIII Rule 3): ";
            for (const auto& seed : selectedSeedsPRIII) {
                std::cout << seed->id << " ";
            }
            std::cout << endl;
        });

        // Measure influence spread
        //double influenceSpreadPMIA = measureInfluenceSpread(networkGraph, selectedSeedsPMIA);
        //double influenceSpreadCELFPP = measureInfluenceSpread(networkGraph, selectedSeedsCELFPP);
        double influenceSpreadPRI = measureInfluenceSpread(networkGraph, selectedSeedsPRI);
        double influenceSpreadPRII = measureInfluenceSpread(networkGraph, selectedSeedsPRII);
        double influenceSpreadPRIII = measureInfluenceSpread(networkGraph, selectedSeedsPRIII);

        // Output the results
        //std::cout << "Influence Spread (PMIA): " << influenceSpreadPMIA << endl;
        //std::cout << "Influence Spread (CELF++): " << influenceSpreadCELFPP << endl;
        std::cout << "Influence Spread (PRI): " << influenceSpreadPRI << endl;
        std::cout << "Influence Spread (PRII): " << influenceSpreadPRII << endl;
        std::cout << "Influence Spread (PRIII): " << influenceSpreadPRIII << endl;

        //std::cout << "Running Time (PMIA): " << runningTimePMIA << " seconds" << endl;
        //std::cout << "Running Time (CELF++): " << runningTimeCELFPP << " seconds" << endl;
        std::cout << "Running Time (PRI): " << runningTimePRI << " seconds" << endl;
        std::cout << "Running Time (PRII): " << runningTimePRII << " seconds" << endl;
        std::cout << "Running Time (PRIII): " << runningTimePRIII << " seconds" << endl;

    } catch (const exception& e) {
        cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
