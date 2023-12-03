/*two types of text files:
type A: only two columns (id, neighbor)
type B: multiple columns (id, checkin time and location info)*/

/*This is an edge-pps updated version*/


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
};

enum class ProbabilityModel 
{
    IC,     // Independent Cascade
    MIA,    // Maximum Influence Arborescence
    PMIA    // Prefix Excluding MIA
};

void loadGeoSocialNetwork (const string& fileName, GeoSocialNetwork& network, ProbabilityModel model)
{
    ifstream file (fileName);
    //cannot open the file
    if (!file)
    {
        throw runtime_error ("Error: Could not open file " + fileName);
    }

    string line;
    istringstream lineStream;

    while (getline (file, line))
    {
        //clear error message
        lineStream.clear ();
        lineStream.str (line);

        int id, neighborId;
        double lat, lon;
        string timestamp;
        string locationId;
        double probability;

        //switch among different models
        switch (model)
        {
            case ProbabilityModel::IC:
                probability = calculatePropagationProbabilityIC ();
                break;
            case ProbabilityModel::MIA:
                probability = calculatePropagationProbabilityMIA ();
                break;
            case ProbabilityModel::PMIA:
                probability = calculatePropagationProbabilityPMIA ();
                break;
            default:
                throw runtime_error ("Unknown probability model");
        }

        //type B
        if (lineStream >> id >> timestamp >> lat >> lon >> locationId)
        {
            network.graph[id] -> addCheckIn (timestamp, lat, lon);
        }
        //type A
        else
        {
            lineStream >> id >> neighborId;
            network.addEdgeWithProbability (id, neighborId, probability);
        }
    }

    file.close ();
}

int main (int argc, char* argv[]) 
{
    //check if txt. file provided in the command line
    if (argc < 2) 
    {
        cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }
    try
    {
        {
            string fileName = argv[1]; //argv[1] is the targeted txt. file and remember its a char array
            GeoSocialNetwork network;
            map<string, ProbabilityModel> modelMap = 
            {
                {"IC", ProbabilityModel::IC},
                {"MIA", ProbabilityModel::MIA},
                {"PMIA", ProbabilityModel::PMIA}
            };

            cout << "Enter model used (IC, MIA, PMIA): ";
            string modelInput;
            cin >> modelInput;
            if (modelMap.find (modelInput) == modelMap.end ()) 
            {
                cerr << "Invalid model type entered.\n";
                return 1;
            }

            ProbabilityModel model = modelMap[modelInput];
            loadGeoSocialNetwork (fileName, network, model);

            for (auto& pair : network.graph)
            {
                cout << "Node " << pair.first << " has neighbors: ";
                for (int neighbor : pair.second -> neighbors)
                {
                    cout << neighbor << " ";
                }
                cout << endl;
                cout << "Check-in data for Node " << pair.first << ":\n";
                for (size_t i = 0; i < pair.second -> checkInTime.size (); ++i)
                {
                    cout << "Check-in time: " << pair.second -> checkInTime[i] << ", Latitude: " << pair.second -> locations[i].first << ", Longitude: " << pair.second -> locations[i].second << "\n";
                }
            } 
        }
    }
    catch(const exception& e)
    {
        cerr << e.what() << '\n';
        return 1;
    }
    
    return 0;
}

double calculatePropagationProbabilityIC ()
{
    //This function calculates the independent cascade model edge pps
}

double calculatePropagationProbabilityMIA ()
{
    //this function calculates the maximum influence arborescence edge pps
}

double calculatePropagationProbabilityPMIA ()
{
    //this function calculates the prefix excluding MIA edge pps
}

void findMIP ()
{
    //implement MIP calculation
}

//functions to calculate MIIA/MIOA
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

double calculateInfluenceSpread(const set<int>& seedSet)
{
    //return influenceSpread
}

//algorithm 1: Greedy algorithm to find the seed nodes for influence maximization
set<int> greedyAlgorithm(int k, const vector<int>& nodes) 
{
    set<int> seedSet;
    for (int i = 0; i < k; ++i) 
    {
        double maxIncrease = 0.0;
        int bestNode = -1;

        for (int node : nodes) 
        {
            if (seedSet.find (node) == seedSet.end ()) 
            {
                set<int> tempSet = seedSet;
                tempSet.insert (node);
                double increase = calculateInfluenceSpread (tempSet) - calculateInfluenceSpread (seedSet);
                if (increase > maxIncrease) 
                {
                    maxIncrease = increase;
                    bestNode = node;
                }
            }
        }

        if (bestNode != -1) 
        {
            seedSet.insert (bestNode);
        }
    }
    return seedSet;
}
//algorithm 2
double calculateActivationProbability (int u, const unordered_set<int>& S, const GeoSocialNetwork& network, unordered_map<int, double>& apCache) 
{
    // Check if we have already calculated ap(u)
    if (apCache.find (u) != apCache.end ()) 
    {
        return apCache[u];
    }

    if (S.find (u) != S.end ()) 
    {
        apCache[u] = 1.0; // Node is in the seed set
    } 
    else 
    {
        const Node* node = network.graph.at (u).get ();
        if (node->neighbors.empty ()) 
        {
            apCache[u] = 0.0; // No in-neighbors
        } 
        else 
        {
            double product = 1.0;
            for (size_t i = 0; i < node->neighbors.size (); ++i) 
            {
                int w = node->neighbors[i];
                double pp = node->edgeProbabilities[i];
                //recursively call the pp calculation
                product *= (1.0 - calculateActivationProbability (w, S, network, apCache) * pp);
            }
            apCache[u] = 1.0 - product;
        }
    }

    return apCache[u];
}
//algorithm 3
double computeAlpha (int v, int u, const GeoSocialNetwork& network, const unordered_set<int>& S, unordered_map<int, double>& apCache, unordered_map<pair<int, int>, double>& alphaCache) 
{
    if (alphaCache.find ({v, u}) != alphaCache.end ()) 
    {
        return alphaCache[{v, u}];
    }

    if (u == v) 
    {
        alphaCache[{v, u}] = 1.0;
    } 
    else 
    {
        Node* nodeU = network.graph.at (u).get ();
        for (auto& neighbor : nodeU->neighbors) 
        {
            if (neighbor == v) 
            {
                double product = 1.0;
                for (auto& inNeighbor : network.graph.at (v)->neighbors) 
                {
                    if (inNeighbor != u) 
                    {
                        product *= (1 - calculateActivationProbability (inNeighbor, S, network, apCache) * network.graph.at (v)->edgeProbabilities[inNeighbor]);
                    }
                }
                alphaCache[{v, u}] = network.graph.at (u)->edgeProbabilities[neighbor] * product;
                break;
            }
        }
    }
    return alphaCache[{v, u}];
}
//algorithm 4: MIA algorithm implementation
unordered_set<int> MIA(const GeoSocialNetwork& network, int k, double theta) 
{
    unordered_set<int> S; // Seed set
    unordered_map<int, double> IncInf; //Incremental Influence
    unordered_map<int, double> apCache; //Cache for activation probabilities
    unordered_map<pair<int, int>, double> alphaCache; //Cache for alpha values

    //Initialization
    for (const auto& pair : network.graph) 
    {
        int v = pair.first;
        IncInf[v] = 0;
        vector<int> miia = MIIA (v, theta, network);

        for (int u : miia) 
        {
            apCache[u] = 0;// Since S is empty
            double alpha = computeAlpha (v, u, network, S, apCache, alphaCache);
            IncInf[u] += alpha * (1 - apCache[u]);
        }
    }

    // Main loop
    for (int i = 0; i < k; ++i) 
    {
        // Find the node with max incremental influence
        int bestNode;
        double maxIncInf = -1;
        for (const auto& pair : IncInf) 
        {
            if (S.find (pair.first) == S.end () && pair.second > maxIncInf) 
            {
                bestNode = pair.first;
                maxIncInf = pair.second;
            }
        }

        S.insert (bestNode);

        // Update Incremental Influence
        for (int v : MIOA (bestNode, theta, network)) 
        {
            if (S.find (v) == S.end ()) 
            {
                for (int w : MIIA(v, theta, network)) 
                {
                    if (S.find (w) == S.end ()) 
                    {
                        IncInf[w] -= computeAlpha (v, w, network, S, apCache, alphaCache) * (1 - apCache[w]);
                        // Recompute ap and alpha
                        apCache[w] = calculateActivationProbability (w, S, network, apCache);
                        IncInf[w] += computeAlpha (v, w, network, S, apCache, alphaCache) * (1 - apCache[w]);
                    }
                }
            }
        }
    }
    return S;
}