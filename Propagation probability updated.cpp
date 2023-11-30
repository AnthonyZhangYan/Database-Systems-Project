/*two types of text files:
type A: only two columns (id, neighbor)
type B: multiple columns (id, checkin time and location info)*/

/*This is an edge-pps updated version*/


#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <map>

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