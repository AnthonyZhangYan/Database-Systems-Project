#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <map>
#include <set>
#include <queue>
#include <cmath>

using namespace std;

class Node
{
    public:
    int nodeId;
    vector<int> neighbors;
    vector<string> checkInTime;
    vector<pair<double, double>> locations;

    vector<double> edgeProbabilities;
    // constructor
    Node (int id) : nodeId (id) {}
    void addNeighbor (int neighborId, double probability)
    {
        neighbors.push_back (neighborId);
        edgeProbabilities.push_back (probability);
    }

    void addCheckIn (const string& time, double lat, double lon)
    {
        checkInTime.push_back (time);
        locations.push_back (make_pair (lat, lon));
    }
};


class GeoSocialNetwork
{
    public:
    unordered_map<int, unique_ptr<Node>> graph;

    //destructor
    ~GeoSocialNetwork () = default;
    //add edges between nodes and their neighbors
    void addEdgeWithProbability (int id, int neighborId, double probability)
    {
        //if id is not in the graph, add it to the graph
        if (graph.find (id) == graph.end ())
        {
            graph[id] = make_unique<Node> (id);
        }
        //point id to its neighbors
        graph[id] -> addNeighbor (neighborId, probability);
    }

    vector<int> getMIOA (int u, double theta) const
    {
        return MIOA (u, theta, *this);
    }

    //influnce spread for IC model
    double getInfluence (int seed, int nodeU) const 
    {
        if (seed == nodeU) 
        {
            return 1.0; // A node always influences itself completely
        }

        double totalInfluence = 0.0;
        unordered_set<int> visited; // Keeps track of visited nodes to avoid cycles
        queue<pair<int, double>> q; // Pair of node and the probability of its activation

        q.push ({seed, 1.0}); // Start from 'seed' node with initial influence of 1
        visited.insert (seed);

        while (!q.empty ()) 
        {
            auto [currentNode, currentProbability] = q.front ();
            q.pop ();

            const Node* node = graph.at(currentNode).get ();
            for (size_t i = 0; i < node->neighbors.size (); ++i) 
            {
                int neighbor = node->neighbors[i];
                double edgeProbability = node->edgeProbabilities[i];
                double newProbability = currentProbability * edgeProbability;

                if (neighbor == nodeU) 
                {
                    totalInfluence += newProbability; // Add to total influence when reaching 'nodeU'
                } 
                else if (visited.find (neighbor) == visited.end ()) 
                {
                    visited.insert (neighbor);
                    q.push ({neighbor, newProbability});
                }
            }
        }
        return totalInfluence;
    }
};

vector<int> MIIA (int v, double theta, const GeoSocialNetwork& network)
//(θ) is a threshold parameter that influences the scope of the influence spread computation
{
    //implementation
    vector<int> S;
    queue<int> Q;
    unordered_map<int, bool> visited;
    Q.push (v);
    visited[v] = true;
    
    while (!Q.empty ()) 
    {
        int u = Q.front ();
        Q.pop ();
        S.push_back (u);
        for (size_t i = 0; i < network.graph.at (u)->neighbors.size (); i++) 
        {
            int w = network.graph.at (u)->neighbors[i];
            double weight = network.graph.at (u)->edgeProbabilities[i];
            if (!visited[w] && weight >= theta) 
            {
                visited[w] = true;
                Q.push (w);
            }
        }
    }
    return S;
}

vector<int> MIOA (int u, double theta, const GeoSocialNetwork& network)
{
    //implementation
    vector<int> S;
    priority_queue<pair<double, int>> Q;
    unordered_map<int, bool> visited;
    Q.push (make_pair(1.0, u));
    visited[u] = true;
    
    while (!Q.empty ()) 
    {
        int v = Q.top ().second;
        double d = Q.top ().first;
        Q.pop ();
        S.push_back (v);
        for (size_t i = 0; i < network.graph.at (v)->neighbors.size (); i++) 
        {
            int w = network.graph.at (v)->neighbors[i];
            double weight = network.graph.at (v)->edgeProbabilities[i];
            if (!visited[w] && weight >= theta) 
            {
                visited[w] = true;
                Q.push (make_pair (d * weight, w));
            }
        }
    }
    return S;
}

double calculateMarginalInfluenceUpperBound (int nodeU, const unordered_set<int>& seedSet, const GeoSocialNetwork& network, double delta, double theta) 
{
    double marginalInfluenceUpperBound = 0.0;
    auto mioaU = network.getMIOA (nodeU, theta);

    unordered_map<int, vector<int>> partitionedMIOA;
    for (int v : mioaU) 
    {
        double influence = network.getInfluence (nodeU, v);
        int partitionIndex = static_cast<int> (log (influence / delta) / log (theta));
        partitionedMIOA[partitionIndex].push_back (v);
    }

    for (const auto& partition : partitionedMIOA) 
    {
        double partitionInfluence = 0.0;
        for (int v : partition.second) 
        {
            partitionInfluence += network.getInfluence (nodeU, v);
        }

        double seedInfluence = 0.0;
        for (int seed : seedSet) 
        {
            seedInfluence += network.getInfluence (seed, nodeU);
        }
        seedInfluence = min (seedInfluence, 1.0);

        marginalInfluenceUpperBound += (1 - seedInfluence) * partitionInfluence;
    }

    return marginalInfluenceUpperBound;
}