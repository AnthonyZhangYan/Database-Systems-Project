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

using namespace std;

// structs used to build the graph
struct TreeNode;
struct Node;
struct Edge;

const int MAX_DEPTH = 100;
// structs for accessing and manipulating nodes in the graph and their properties
struct MIP 
{
    vector<Node*> path;
    double propagationProbability;

    MIP () : propagationProbability (0.0) {}

    MIP (const MIP& other) : path (other.path), propagationProbability (other.propagationProbability) {}
};

struct Node 
{
    int id;
    unordered_set<Node*> inNeighbors;
    unordered_set<Node*> outNeighbors;
    vector<Edge> outgoingEdges;
    vector<Edge> incomingEdges;
    // constructor
    Node (int id) : id (id) {}
    // function for calculating outgoing probability for nodes
    double calculateOutgoingProbability () 
    {
        return outNeighbors.empty () ? 0.0 : 1.0 / outNeighbors.size ();
    }
};

struct Edge 
{
    Node* from;
    Node* to;
    double probability;
    // constructor
    Edge (Node* fromNode, Node* toNode, double prob) : from (fromNode), to (toNode), probability (prob) {}
};

// tree data structure building
struct TreeNode : public std::enable_shared_from_this<TreeNode> 
{
    shared_ptr<Node> graphNode;
    weak_ptr<TreeNode> parent;
    vector<shared_ptr<TreeNode>> children;

    TreeNode (shared_ptr<Node> graphNode) : graphNode (graphNode) {}

    static shared_ptr<TreeNode> Create (shared_ptr<Node> node) 
    {
        return make_shared<TreeNode> (node);
    }

    // DFA algorithm to traverse the tree
    void DFS () 
    {
        cout << "Node ID: " << graphNode->id << endl;
        for (auto& child : children) {
            child->DFS();
        }
    }
    // getter function for nodes
    vector<shared_ptr<Node>> getAllNodes () 
    {
        vector<shared_ptr<Node>> nodes;
        // use a stack to implement the getter function
        stack<shared_ptr<TreeNode>> stack;
        stack.push (shared_from_this ());
        while (!stack.empty ()) 
        {
            auto currentNode = stack.top ();
            stack.pop ();
            nodes.push_back (currentNode->graphNode);
            for (auto& child : currentNode->children) 
            {
                stack.push (child);
            }
        }
        return nodes;
    }
};

struct CompareProb 
{
    // operator overload of () to compare probabilities of different edges
    bool operator() (const pair<double, Node*>& a, const pair<double, Node*>& b) 
    {
        return a.first < b.first;
    }
};

// Graph class which is used to build our social network
class Graph 
{
public:
    unordered_map<int, shared_ptr<Node>> nodes;

    //default constructor
    Graph() = default;
    
    // function for adding edges between nodes
    void addEdge(int fromId, int toId) 
    {
        // if id is not found in the neighbors list, add an edge
        if (nodes.find (fromId) == nodes.end ()) 
        {
            nodes[fromId] = make_shared<Node> (fromId);
        }
        if (nodes.find (toId) == nodes.end ()) 
        {
            nodes[toId] = make_shared<Node> (toId);
        }

        // declaration of ingoing and outgoing neighbors and assign their properties
        Node* fromNode = nodes[fromId].get ();
        Node* toNode = nodes[toId].get ();

        fromNode->outNeighbors.insert (toNode);
        toNode->inNeighbors.insert (fromNode);
        //double probability = fromNode->calculateOutgoingProbability();
        double probability = 0.6;
        fromNode->outgoingEdges.emplace_back (fromNode, toNode, probability);
        toNode->incomingEdges.emplace_back (fromNode, toNode, probability);
    }

    // load nodes information from files and build edges according to the rules
    void readEdgesFromFile (const string& filename) 
    {
        ifstream file (filename);
        int fromId, toId;

        while (file >> fromId >> toId) 
        {
            addEdge (fromId, toId);
        }
    }

    Node* getNodeById (int id) 
    {
        return nodes[id].get ();
    }

    // getter functions for edges
    vector<Edge> getIncomingEdges (int nodeId) 
    {
        if (nodes.find (nodeId) == nodes.end ()) 
        {
            return vector<Edge> ();
        }
        return nodes[nodeId]->incomingEdges;
    }

    vector<Edge> getOutgoingEdges (int nodeId) 
    {
        if (nodes.find (nodeId) == nodes.end ()) 
        {
            return vector<Edge> ();
        }
        return nodes[nodeId]->outgoingEdges;
    }

    // MIP calculation
    pair<MIP*, unordered_map<Node*, Node*>> CalculateMIP (Node* u, Node* v) 
    {
        unordered_map<Node*, double> MaxProb;
        unordered_map<Node*, Node*> Predecessor;
        // use a priority queue to do the calculation
        priority_queue<pair<double, Node*>, vector<pair<double, Node*>>, CompareProb> PriorityQueue;

        for (auto& pair : nodes) 
        {
            MaxProb[pair.second.get ()] = (pair.second.get () == u) ? 1.0 : 0.0;
            Predecessor[pair.second.get ()] = nullptr;
        }

        PriorityQueue.push (make_pair (1.0, u));

        while (!PriorityQueue.empty ()) 
        {
            Node* CurrentNode = PriorityQueue.top ().second;
            PriorityQueue.pop ();

            if (CurrentNode == v) 
                break;

            for (Edge& edge : CurrentNode->outgoingEdges) 
            {
                Node* Neighbor = edge.to;
                double NewProb = MaxProb[CurrentNode] * edge.probability;

                if (NewProb > MaxProb[Neighbor]) 
                {
                    MaxProb[Neighbor] = NewProb;
                    Predecessor[Neighbor] = CurrentNode;
                    PriorityQueue.push (make_pair (NewProb, Neighbor));
                }
            }
        }

        // maximization of the propagation probability
        MIP* mip = new MIP();
        Node* CurrentNode = v;
        mip->propagationProbability = MaxProb[v];

        while (CurrentNode != nullptr && CurrentNode != u) 
        {
            mip->path.push_back (CurrentNode);
            CurrentNode = Predecessor[CurrentNode];
        }
        mip->path.push_back (u);
        reverse(mip->path.begin (), mip->path.end ());
        return {mip, Predecessor};
    }

    // MIIA and MIOA calculations
    shared_ptr<TreeNode> calculateMIIA (int nodeId, double theta) 
    {
        // if node is not found, return empty tree
        if (nodes.find (nodeId) == nodes.end ()) 
            return nullptr;

        shared_ptr<Node> target = nodes[nodeId];
        auto root = make_shared<TreeNode> (target);
        unordered_map<Node*, shared_ptr<TreeNode>> createdNodes;
        createdNodes[target.get ()] = root;

        for (auto& pair : nodes) 
        {
            shared_ptr<Node> start = pair.second;
            if (start == target) 
                continue;

            auto mipAndPredecessors = CalculateMIP (start.get (), target.get ());
            MIP* mip = mipAndPredecessors.first;

            if (mip->propagationProbability >= theta) 
            {
                shared_ptr<TreeNode> lastAddedNode = root;

                for (auto it = mip->path.rbegin (); it != mip->path.rend (); ++it) 
                {
                    Node* node = *it;
                    if (node == target.get () || createdNodes.find (node) != createdNodes.end ()) 
                        continue;

                    auto newNode = make_shared<TreeNode> (nodes[node->id]);
                    lastAddedNode->children.push_back (newNode);
                    newNode->parent = lastAddedNode;
                    createdNodes[node] = newNode;
                    lastAddedNode = newNode;
                }
            }
            delete mip;
        }
        return root;
    }

    shared_ptr<TreeNode> calculateMIOA (int nodeId, double theta) 
    {
        if (nodes.find (nodeId) == nodes.end ()) 
            return nullptr;

        shared_ptr<Node> source = nodes[nodeId];
        auto root = make_shared<TreeNode> (source);
        unordered_map<Node*, shared_ptr<TreeNode>> createdNodes;
        createdNodes[source.get ()] = root;

        for (auto& pair : nodes) 
        {
            shared_ptr<Node> target = pair.second;
            if (source == target) 
                continue;

            auto mipAndPredecessors = CalculateMIP (source.get (), target.get ());
            MIP* mip = mipAndPredecessors.first;

            if (mip->propagationProbability >= theta) 
            {
                shared_ptr<TreeNode> lastAddedNode = root;

                for (Node* node : mip->path) 
                {
                    if (node == source.get () || createdNodes.find (node) != createdNodes.end ()) 
                        continue;

                    auto newNode = make_shared<TreeNode> (nodes[node->id]);
                    lastAddedNode->children.push_back (newNode);
                    newNode->parent = lastAddedNode;
                    createdNodes[node] = newNode;
                    lastAddedNode = newNode;
                }
            }
            delete mip;
        }
        return root;
    }

    // ... Other methods of the Graph class ...
    // getter fuuntions to access neighbors (in, out, all)
    unordered_set<Node*> getAllNeighbors (int nodeId) 
    {
        unordered_set<Node*> neighbors;
        if (nodes.find (nodeId) != nodes.end ()) 
        {
            shared_ptr<Node> node = nodes.at (nodeId);

            for (Node* outNeighbor : node->outNeighbors) 
            {
                neighbors.insert (outNeighbor);
            }
            for (Node* inNeighbor : node->inNeighbors) 
            {
                neighbors.insert (inNeighbor);
            }
        }
        return neighbors;
    }

    const unordered_set<Node*>& getOutNeighbors (int nodeId) 
    {
        static unordered_set<Node*> emptySet; // To handle the case where nodeId is not found

        if (nodes.find (nodeId) != nodes.end ()) 
        {
            return nodes.at (nodeId)->outNeighbors;
        } 
        else 
        {
            return emptySet;
        }
    }

    const unordered_set<Node*>& getInNeighbors (int nodeId) 
    {
        static unordered_set<Node*> emptySet; // To handle the case where nodeId is not found

        if (nodes.find (nodeId) != nodes.end ()) 
        {
            return nodes.at (nodeId)->inNeighbors;
        } 
        else 
        {
            return emptySet;
        }
    }

    // print function for debugging purposes
    void printGraph () 
    {
        for (const auto& pair : nodes) 
        {
            shared_ptr<Node> node = pair.second;
            if (node) 
            {
                cout << "Node " << node->id << ":\n";
                cout << "  In-Neighbors: ";
                for (const auto& inNeighbor : node->inNeighbors) 
                {
                    cout << inNeighbor->id << " ";
                }
                cout << "\n  Out-Neighbors: ";
                for (const auto& outNeighbor : node->outNeighbors) 
                {
                    cout << outNeighbor->id << " ";
                }
                cout << "\n";
            }
        }
    }

    // helper function for finding MIIA and MIOA
    void printMIPsFromTree(const shared_ptr<TreeNode>& node, vector<int>& currentPath) 
    {
        if (!node) 
            return;

        // Add the current node to the path
        currentPath.push_back(node->graphNode->id);

        if (node->children.empty ()) 
        {
            // If it's a leaf node, print the path from this node to the root
            cout << "MIP: ";
            for (auto it = currentPath.rbegin (); it != currentPath.rend (); ++it) 
            {
                cout << *it << " ";
            }
            cout << endl;
        } 
        else 
        {
            // Continue to traverse for each child
            for (const auto& child : node->children) 
            {
                printMIPsFromTree (child, currentPath);
            }
        }

        // Remove the current node before going back up the tree
        currentPath.pop_back ();
    }

    void findAllMIPsInMIIA (const shared_ptr<TreeNode>& root) 
    {
        vector<int> currentPath;
        printMIPsFromTree (root, currentPath);
    }

    void printMIPsInMIOA (const shared_ptr<TreeNode>& node, vector<int>& currentPath) 
    {
        if (!node)
             return;

        // Add the current node to the path
        currentPath.push_back(node->graphNode->id);

        if (node->children.empty ()) 
        {
            // If it's a leaf node, print the path from the root to this node
            cout << "MIP: ";
            for (int id : currentPath) 
            {
                cout << id << " ";
            }
            cout << endl;
        } 
        else 
        {
            // Continue to traverse for each child
            for (const auto& child : node->children) 
            {
                printMIPsInMIOA (child, currentPath);
            }
        }

        // Remove the current node before going back up the tree
        currentPath.pop_back ();
    }

    void findAllMIPsInMIOA (const shared_ptr<TreeNode>& root) 
    {
        vector<int> currentPath;
        printMIPsInMIOA (root, currentPath);
    }

    // find the propagation probability for edges
    double findPropagationProbability (Node* w, Node* u) 
    {
        // if no neighbors, probability is automatically 0.0
        if (!w) 
            return 0.0;  // Check for null pointer
        for (auto& edge : w->outgoingEdges) 
        {
            if (edge.to == u) 
            {
                return edge.probability;
            }
        }
        return 0.0;  // Return 0 if no edge is found
    }

    void calculateAPIteratively (shared_ptr<TreeNode> root, const unordered_set<int>& S, double theta, unordered_map<int, double>& alphaCache) 
    {
        // similar if no neighbors, end the function
        if (!root) 
            return;

        // Initialize all node activation probabilities to zero
        for (auto& pair : nodes) 
        {
            alphaCache[pair.first] = (S.find (pair.first) != S.end ()) ? 1.0 : 0.0;
        }

        // Post-order traversal stack
        stack<shared_ptr<TreeNode>> postOrderStack;
        stack<shared_ptr<TreeNode>> stack;
        stack.push (root);
        unordered_set<shared_ptr<TreeNode>> visited;

        while (!stack.empty ()) 
        {
            shared_ptr<TreeNode> node = stack.top ();
            stack.pop ();
            postOrderStack.push (node);
            
            for (auto& child : node->children) 
            {
                if (visited.find (child) == visited.end ()) 
                {
                    stack.push (child);
                    visited.insert (child);
                }
            }
        }

        // Calculate activation probabilities in post-order
        while (!postOrderStack.empty ()) 
        {
            shared_ptr<TreeNode> treeNode = postOrderStack.top ();
            postOrderStack.pop ();
            shared_ptr<Node> graphNode = treeNode->graphNode;

            if (S.find (graphNode->id) != S.end ()) 
                continue; // Skip seed nodes

            double product = 1.0;
            for (auto& neighbor : graphNode->inNeighbors) 
            {
                double pp_wu = findPropagationProbability (neighbor, graphNode.get ());
                double ap_w = alphaCache[neighbor->id];
                product *= (1 - ap_w * pp_wu);
            }

            alphaCache[graphNode->id] = 1 - product;
        }
    }

    shared_ptr<Node> getNode (int nodeId) 
    {
        if (nodes.find (nodeId) != nodes.end ()) 
        {
            return nodes[nodeId];
        } 
        else 
        {
            return nullptr;  // Return nullptr if node does not exist
        }
    }
    
    // alpha algorithm 3
    void calculateAlphaIterativelyV3 (shared_ptr<TreeNode> root, const unordered_set<int>& S, double theta, unordered_map<int, double>& alphaCache) 
    {
        if (!root) 
            return;

        // Initialize all node influence values to zero, except for seed nodes
        for (auto& pair : nodes) 
        {
            alphaCache[pair.first] = (S.find (pair.first) != S.end ()) ? 1.0 : 0.0;
        }

        // BFS to traverse MIIA tree from root to leaves
        queue<shared_ptr<TreeNode>> q;
        q.push(root);

        while (!q.empty ()) 
        {
            shared_ptr<TreeNode> treeNode = q.front ();
            q.pop ();
            shared_ptr<Node> graphNode = treeNode->graphNode;

            // Skip seed nodes
            if (S.find (graphNode->id) != S.end ()) 
                continue;

            double alpha = 0.0;
            if (!graphNode->inNeighbors.empty ()) 
            {
                // Compute influence value for this node
                for (auto& inNeighbor : graphNode->inNeighbors) 
                {
                    double pp_wu = findPropagationProbability (inNeighbor, graphNode.get ());
                    alpha += alphaCache[inNeighbor->id] * pp_wu;
                }
            }

            // Update influence value for the node
            alphaCache[graphNode->id] = alpha;

            // Add children to queue for BFS
            for (auto& child : treeNode->children) 
            {
                q.push (child);
            }
        }
    }

    // set of MIA
    unordered_set<int> MIA (int k, double theta)
    {
        unordered_set<int> S;
        unordered_map<int, double> IncInf;
        unordered_map<int, double> apCache;
        unordered_map<int, double> alphaCache;

        // Initialization
        for (auto& pair : nodes) 
        {
            int v = pair.first;
            IncInf[v] = 0;

            auto miiaRoot = calculateMIIA (v, theta);
            auto mioaRoot = calculateMIOA (v, theta);

            calculateAPIteratively (miiaRoot, S, theta, apCache); // Alg. 2
            calculateAlphaIterativelyV3 (miiaRoot, S, theta, alphaCache); // Alg. 3

            for (auto& nodePair : nodes) 
            {
                int u = nodePair.first;
                if (miiaRoot->graphNode->id == u) 
                {
                    IncInf[u] += alphaCache[v] * (1 - apCache[u]);
                }
            }
        }

        // Main loop
        for (int i = 0; i < k; ++i) 
        {
            // Find u with maximum IncInf
            int u_max = -1;
            double maxInf = -1;
            for (auto& pair : IncInf) 
            {
                if (S.find (pair.first) == S.end () && pair.second > maxInf) 
                {
                    maxInf = pair.second;
                    u_max = pair.first;
                }
            }

            // Update incremental influence spreads
            auto mioaRootU = calculateMIOA (u_max, theta);
            for (auto& v : mioaRootU->children) 
            { // For each v in MIOA(u, θ) \ S
                auto miiaRootV = calculateMIIA (v->graphNode->id, theta);
                for (auto& w : miiaRootV->children) 
                { // For each w in MIIA(v, θ) \ S
                    IncInf[w->graphNode->id] -= alphaCache[v->graphNode->id] * (1 - apCache[w->graphNode->id]);
                }
            }

            S.insert (u_max);

            // Recompute ap and alpha for affected nodes
            for (auto& v : mioaRootU->children) 
            { // For each v in MIOA(u, θ) \ S
                auto miiaRootV = calculateMIIA (v->graphNode->id, theta);
                calculateAPIteratively (miiaRootV, S, theta, apCache); // Alg. 2
                calculateAlphaIterativelyV3 (miiaRootV, S, theta, alphaCache); // Alg. 3

                for (auto& w : miiaRootV->children) 
                { // For each w in MIIA(v, θ) \ S
                    IncInf[w->graphNode->id] += alphaCache[v->graphNode->id] * (1 - apCache[w->graphNode->id]);
                }
            }
        }
        return S;
    }
};

int main () 
{
    Graph graph;

    // Create a larger graph:
    // Example connections (directed edges):
    // 1 -> 2, 3
    // 2 -> 4, 5
    // 3 -> 6
    // 4 -> 7
    // 5 -> 7, 8
    // 6 -> 7
    graph.addEdge (1, 2);
    graph.addEdge (1, 3);
    graph.addEdge (2, 4);
    graph.addEdge (2, 5);
    graph.addEdge (3, 6);
    graph.addEdge (4, 7);
    graph.addEdge (5, 7);
    graph.addEdge (5, 8);
    graph.addEdge (6, 7);
    graph.addEdge (1, 7);
    graph.addEdge (7, 3);
    graph.addEdge (3, 1);
    graph.addEdge (7, 5);
    graph.addEdge (5, 2);
    graph.addEdge (2, 1);
    graph.addEdge (8, 2);

    // Test printing the graph
    cout << "Graph structure:\n";
    graph.printGraph ();

    /*
    // Testing CalculateMIP from Node 1 to Node 7
    Node* start = graph.getNodeById (7);
    Node* end = graph.getNodeById (1);
    cout<< "Main start and end " << start << ' ' << end;
    //MIP* mip = graph.CalculateMIP (start, end);
    auto mipAndPredecessors = graph.CalculateMIP (start,end);
    MIP* mip = mipAndPredecessors.first;
    auto& Predecessor = mipAndPredecessors.second;
    cout << "\nMaximum Influence Path (MIP) from Node 1 to Node 7:\n";
    for (Node* node : mip->path) 
    {
        cout << node->id << " ";
    }
    cout << "\nPropagation Probability: " << mip->propagationProbability << "\n";
    delete mip; // Clean up the dynamically allocated MIP

    // Test calculateMIIA
    double theta = 0.01; // Example threshold
    cout << "\nMIIA Tree (DFS Traversal) for Node 7:\n";
    shared_ptr<TreeNode> miiaRoot = graph.calculateMIIA (7, theta);
    if (miiaRoot) 
    {
        miiaRoot->DFS ();
        cout << "Children" ;
    } 
    else 
    {
        cout << "MIIA Tree not found for Node 7.\n";
    }

    // Test calculateMIOA
    cout << "\nMIOA Tree (DFS Traversal) for Node 1:\n";
    shared_ptr<TreeNode> mioaRoot = graph.calculateMIOA (1, theta);
    if (mioaRoot) 
    {
        mioaRoot->DFS ();
    } 
    else 
    {
        cout << "MIOA Tree not found for Node 1.\n";
    }

    cout <<"MIIA :\n";
    shared_ptr<TreeNode> miiaRoot2 = graph.calculateMIIA (7, theta);
    graph.findAllMIPsInMIIA (miiaRoot2);

    cout <<"MIOA :\n";
    shared_ptr<TreeNode> mioaRoot2= graph.calculateMIOA (1, theta);
    graph.findAllMIPsInMIOA (mioaRoot2);

    // Define the seed set
    unordered_set<int> S = {1, 3};

    // Set theta threshold
    double theta = 0.01;

    // Initialize a cache for storing activation probabilities
    unordered_map<int, double> alphaCache;

    try {
        // Calculate and print activation probabilities for all nodes
        cout << "Activation probabilities:" << endl;
        for (const auto& pair : graph.nodes) 
        {
            shared_ptr<Node> node = pair.second;
            if (node) 
            {
                double alpha = graph.calculateAlpha (node, S, theta, alphaCache);
                cout << "Node " << node->id << ": " << alpha << endl;
            }
        }
    } 
    catch (const std::exception& e) 
    {
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

    unordered_set<int> seedSet = graph.MIA (k, theta);

    cout << "Seed set: ";
    for (int node : seedSet) 
    {
        cout << node << " ";
    }
    cout << endl;

    return 0;
}