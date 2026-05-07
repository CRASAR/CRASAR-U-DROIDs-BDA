import labelbox as lb
import json
import argparse

parser = argparse.ArgumentParser(description='Upload Labels to Labelbox.')
parser.add_argument('--api_key_file', help="The path to the file containing the API key", type=str)
parser.add_argument('--project_id_file', help="The project ID for this task on labelbox", type=str)
args = parser.parse_args()

#Load the API key
f = open(args.api_key_file)
API_KEY = f.readlines()[0].replace(" ", "").replace("\n", "").replace("\r", "")
f.close()
client = lb.Client(API_KEY)

f = open(args.project_id_file)
PROJECT_ID_MAP = json.loads(f.read())
f.close()

############################################################
############## CREATE RDA ONTOLOGY #########################
############################################################

rda_project = client.get_project(project_id = PROJECT_ID_MAP["sUAS Overhead RDA"])
rda_ontology_builder = lb.OntologyBuilder(tools=[lb.Tool(name="Road Line",                                 tool=lb.Tool.Type.LINE),
												 lb.Tool(name="Passable with Difficulty (Flooding)",       tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Passable with Difficulty (Obstructions)",   tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Passable with Difficulty (Road Condition)", tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Not Passable (Destruction)",                tool=lb.Tool.Type.POLYGON), 
												 lb.Tool(name="Not Passable (Obstructions)",               tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Not Passable (Flooding)",                   tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Not Able To Determine",                     tool=lb.Tool.Type.POLYGON)])
rda_ontology = client.create_ontology("TAMU RDA", rda_ontology_builder.asdict())
try:
	rda_project.setup_editor(rda_ontology)
except lb.exceptions.ResourceConflict:
	print("Error: Attempting to set up an editor on a project that already has an editor defined.\nSkipping this project: sUAS Overhead RDA")

############################################################
############## CREATE BDA ONTOLOGY #########################
############################################################

bda_project = client.get_project(project_id = PROJECT_ID_MAP["sUAS Overhead BDA"])
bda_ontology_builder = lb.OntologyBuilder(tools=[lb.Tool(name="No Damage OR Very Minor Damage", tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Minor Damage",                   tool=lb.Tool.Type.POLYGON), 
												 lb.Tool(name="Moderate Damage",                tool=lb.Tool.Type.POLYGON), 
												 lb.Tool(name="Severe Damage",                  tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Total Destruction",              tool=lb.Tool.Type.POLYGON),
												 lb.Tool(name="Not Able To Determine",          tool=lb.Tool.Type.POLYGON)])
bda_ontology = client.create_ontology("sUAS HAZUS BDA", bda_ontology_builder.asdict())
try:
	bda_project.setup_editor(bda_ontology)
except lb.exceptions.ResourceConflict:
	print("Error: Attempting to set up an editor on a project that already has an editor defined.\nSkipping this project: sUAS Overhead BDA")

############################################################
############## CREATE DEBRIS ONTOLOGY ######################
############################################################

debris_project = client.get_project(project_id = PROJECT_ID_MAP["sUAS Overhead Debris Assessment"])
debris_ontology_builder = lb.OntologyBuilder(tools=[lb.Tool(name="Traditional Construction Debris", tool=lb.Tool.Type.POLYGON),
													lb.Tool(name="Reinforced Construction Debris",  tool=lb.Tool.Type.POLYGON), 
													lb.Tool(name="Vegetative Debris",               tool=lb.Tool.Type.POLYGON),
													lb.Tool(name="Not Able To Determine",           tool=lb.Tool.Type.POLYGON),])
debris_ontology = client.create_ontology("sUAS HAZUS Debris", debris_ontology_builder.asdict())
try:
	debris_project.setup_editor(debris_ontology)
except lb.exceptions.ResourceConflict:
	print("Error: Attempting to set up an editor on a project that already has an editor defined.\nSkipping this project: sUAS Overhead Debris Assessment")

############################################################
############## CREATE BUILDING TYPE ONTOLOGY ###############
############################################################

building_type_project = client.get_project(project_id = PROJECT_ID_MAP["sUAS Overhead Building Type Categorization"])
building_type_ontology_builder = lb.OntologyBuilder(tools=[lb.Tool(name="Manufactured/Pre-fabricated", tool=lb.Tool.Type.POLYGON),
														   lb.Tool(name="Traditional Construction",    tool=lb.Tool.Type.POLYGON),
														   lb.Tool(name="Reinforced Construction",     tool=lb.Tool.Type.POLYGON),
														   lb.Tool(name="Not Able To Determine",       tool=lb.Tool.Type.POLYGON)])
building_type_ontology = client.create_ontology("sUAS HAZUS Building Type", debris_ontology_builder.asdict())
try:
	debris_project.setup_editor(building_type_ontology)
except lb.exceptions.ResourceConflict:
	print("Error: Attempting to set up an editor on a project that already has an editor defined.\nSkipping this project: sUAS Overhead Building Type Categorization")