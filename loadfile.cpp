#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

class Node
{
    public:
    int nodeId;
    vector<int> neighbors;
    // constructor
    Node (int id) : nodeId (id) {}
    void addNeighbor (int neighborId)
    {
        neighbors.push_back (neighborId);
    }
};

class GeoSocialNetwork
{
    public:
    unordered_map <int, Node*> graph;

    //destructor
    ~GeoSocialNetwork ()
    {
        for (auto& pair : graph)
        {
            delete pair.second;
        }
    }
    //add edges between nodes and their neighbors
    void addEdge (int id, int neighborId)
    {
        //if id is not in the graph, add it to the graph
        if (graph.find (id) == graph.end ())
        {
            graph[id] = new Node (id);
        }
        //point id to its neighbors
        graph[id] -> addNeighbor (neighborId);
    }
};

void loadGeoSocialNetwork (const string& fileName, GeoSocialNetwork& network)
{
    ifstream file (fileName);
    //cannot open the file
    if (!file)
    {
        cerr << "Error: Could not open file " << fileName << endl;
        exit (1);
    }

    string line;
    int id, neighborId;

    while (getline (file, line))
    {
        istringstream istring (line);
        //read id and neighborId in sequence ignoring whitespace
        if (istring >> id >> neighborId)
        {
            network.addEdge (id, neighborId);
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

    string fileName = argv[1]; //argv[1] is the targeted txt. file and remember its a char array
    GeoSocialNetwork network;
    loadGeoSocialNetwork (fileName, network);

    for (auto& pair : network.graph)
    {
        cout << "Node " << pair.first << " has neighbors: ";
        for (int neighbor : pair.second -> neighbors)
        {
            cout << neighbor << " ";
        }
        cout << endl;
    }
    return 0;
}
