/*two types of text files:
type A: only two columns (id, neighbor)
type B: multiple columns (id, checkin time and location info)*/


#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

using namespace std;

class Node
{
    public:
    int nodeId;
    vector<int> neighbors;
    vector<string> checkInTime;
    vector<pair<double, double>> locations;
    // constructor
    Node (int id) : nodeId (id) {}
    void addNeighbor (int neighborId)
    {
        neighbors.push_back (neighborId);
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
    void addEdge (int id, int neighborId)
    {
        //if id is not in the graph, add it to the graph
        if (graph.find (id) == graph.end ())
        {
            graph[id] = make_unique<Node> (id);
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
        //type B
        if (lineStream >> id >> timestamp >> lat >> lon >> locationId)
        {
            network.graph[id] -> addCheckIn (timestamp, lat, lon);
        }
        //type A
        else
        {
            lineStream >> id >> neighborId;
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
    try
    {
        {
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
