import os

import pytest
from sklearn.metrics import accuracy_score, classification_report

from supervised_multilabel_classifier import core
from supervised_multilabel_classifier.service.model_loader import load_model
from supervised_multilabel_classifier.service.csv_loader import CsvLoader


@pytest.fixture
def test_fixture():
    model = load_model()

    x_vec = core.AweVectorizer(model)
    y_vec = core.MultiLabelVectorizer()

    pytest.x_vec, pytest.y_vec = x_vec, y_vec


def get_file_path(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_set/' + filename))


def test_model_sample(test_fixture):
    test_model(get_file_path('training_data.csv'))


def test_model_imdb(test_fixture):
    test_model(get_file_path('imdb_dataset.csv'))


def test_model(filename):
    data_loader = CsvLoader(filename, ["ID", "Text", "Category"])
    x_train, x_test, y_train, y_true, vec2ids = data_loader.get_vectors(pytest.x_vec, pytest.y_vec)

    predictor = core.Predictor()
    predictor.fit(x_train, y_train)
    y_predicted = predictor.predict(x_test)
    accuracy = accuracy_score(y_true, y_predicted)
    print(classification_report(y_true, y_predicted, target_names=pytest.y_vec.get_classes()))
    assert (accuracy > 0.5)


def test_prediction_sample(test_fixture):
    y_test = ["ny"]
    text = "New York is a lovely city although it is not a capital"
    assert_prediction(text, y_test, get_file_path('training_data.csv'))

    y_test = ["london", "paris"]
    text = "I am going to visit London and Paris"
    assert_prediction(text, y_test, get_file_path('training_data.csv'))


def test_prediction_imdb(test_fixture):
    y_test = ["Action"]
    mad_max_fury_road = '''Following an energy crisis, the world has become a desert wasteland and civilization has 
    collapsed. Max Rockatansky, a survivor, is captured by the War Boys, the army of the tyrannical Immortan Joe, 
    and taken to Joe's Citadel. Designated a universal blood donor, Max is imprisoned and used as a "blood bag" for a 
    sick War Boy called Nux. Meanwhile, Imperator Furiosa, one of Joe's lieutenants, is sent in her armoured 
    semi-truck, the "War Rig", to collect gasoline and ammunition. When she drives off-route, Joe realizes that his 
    five wives—women selected for breeding—are missing, and fleeing with her. Joe leads his entire army in pursuit of 
    Furiosa, calling on the aid of nearby Gas Town and the Bullet Farm. 

    Nux joins the pursuit with Max strapped to his car to continue supplying blood. A battle ensues between the War Rig 
    and Joe's forces. Furiosa drives into a sand storm, evading her pursuers, except Nux, who attempts to sacrifice 
    himself to destroy the Rig. Max escapes and restrains Nux, but the car is destroyed. After the storm, 
    Max finds Furiosa repairing the Rig, accompanied by the wives: Capable, Cheedo, Toast, the Dag, and the Splendid 
    Angharad, who is heavily pregnant with Joe's child. Max steals the Rig, but Furiosa has activated a kill switch which 
    disables it. Max reluctantly agrees to let Furiosa and the wives accompany him; Nux climbs on the Rig as it leaves 
    and attempts to kill Furiosa, but is overcome and thrown out, and is picked up by Joe's army. 
    
    Furiosa drives through a biker gang-controlled canyon having bartered a deal for safe passage. However, with Joe's 
    forces pursuing, the gang turns on her, forcing her and the group to flee, while the bikers detonate the canyon walls 
    to block Joe. Max and Furiosa fight pursuing bikers as Joe's car, with Nux now on board, breaks through the blockade 
    and eventually attacks the War Rig, allowing Nux to board and once again attempts to attack Furiosa, but fails and 
    falls to the disappointment of Joe. However, as the Rig escapes, Angharad also falls off trying to help Max and is 
    run over by Joe's car, killing her and her child. Furiosa explains to Max that they are escaping to the "Green 
    Place", an idyllic land she remembers from her childhood. Capable finds Nux hiding in the Rig, distraught over his 
    failure, and consoles him. That night, the Rig gets stuck in the mud. Furiosa and Max slow Joe's forces with mines, 
    but Joe's ally, the Bullet Farmer, continues pursuing them. Nux helps Max free the Rig while Furiosa shoots and 
    blinds the Bullet Farmer. Max walks into the dark to confront the Bullet Farmer and his men, wordlessly returning 
    with guns and ammunition and covered in blood. 
    
    They drive the War Rig overnight through swampland and desert, coming across a naked woman the next day. Max suspects 
    a trap, though Furiosa approaches the woman and states her history and clan affiliation. The naked woman summons her 
    clan, the Vuvalini, who recognize Furiosa as one of their own who was kidnapped as a child. Furiosa is devastated to 
    learn that the swampland they passed was indeed the Green Place, now uninhabitable. The group then plans to ride 
    motorbikes across immense salt flats in the hope of finding a new home. Max chooses to stay behind, but after seeing 
    visions of a child he failed to save, he convinces them to return to the undefended Citadel, which has ample water 
    and greenery that Joe keeps for himself, and trap Joe and his army in the bikers' canyon. 
    
    The group heads back to the Citadel, but they are attacked en route by Joe's forces, and Furiosa is seriously 
    wounded. Joe positions his car in front of the War Rig to slow it, while Max fights Joe's giant son, Rictus Erectus. 
    Joe captures Toast, who manages to distract him long enough for Furiosa to kill him. Nux sacrifices himself by 
    wrecking the Rig, killing Rictus and blocking the canyon, allowing Max, Furiosa, the wives, and the surviving 
    Vuvalini to escape in Joe's car, where Max transfuses his blood to Furiosa, saving her life. 
    
    At the Citadel, the impoverished citizens react to Joe's death with joy. Furiosa, the wives, and the Vuvalini are 
    cheered by the people and welcomed by the remaining War Boys. Max shares a respectful glance with Furiosa before 
    blending into the crowd and departing for parts unknown. '''

    assert_prediction(mad_max_fury_road, y_test, get_file_path('imdb_dataset.csv'))

    y_test = ["Horror"]
    the_evil_dead = '''Five Michigan State University students—Ash Williams, his girlfriend, Linda; Ash's sister, 
    Cheryl; their friend Scott; and his girlfriend Shelly—are vacationing at an isolated cabin in rural Tennessee. On 
    their first night there, Cheryl hears a faint, demonic voice telling her to "join us" just before her hand seems 
    to become possessed and draw a picture of a book with a deformed face on its cover. Shaken and confused, 
    she decides not to mention the incident to the others. 

    When the trapdoor to the cellar mysteriously flies open during dinner, Ash and Scott go down to investigate and find 
    the Naturan Demanto, a Sumerian version of the Egyptian Book of the Dead, along with a tape recorder belonging to the 
    archaeologist who owned it. Scott plays a tape of the archaeologist reciting a series of incantations that resurrect 
    a mysterious, demonic entity. Agitated, Cheryl goes outside to investigate strange noises she hears. In the woods, 
    she is raped by demonically possessed trees. When she returns to the cabin bruised and anguished, Ash agrees to take 
    her into town for the night. However, they soon discovers that the bridge to the cabin has been destroyed. Back at 
    the cabin, Ash listens to more of the tape, learning that the only way to kill the entity is to dismember the body 
    when it possesses a host. Cheryl succumbs to the entity and attacks the others, stabbing Linda in the ankle with a 
    pencil before Scott is able to lock her in the cellar. 
    
    Shelly becomes possessed as well, forcing Scott to chop up her body with an axe and bury the remains. Shaken by the 
    experience, he leaves to find a way back to town but soon returns mortally wounded; he dies while warning Ash that 
    the trees will not let them escape alive. When Ash goes to check on Linda, he is horrified to find that she too has 
    become possessed by the demon. She attacks him, but he stabs her with a Sumerian dagger. Unwilling to dismember her, 
    he buries her instead. But when she revives and attacks him, he is forced to decapitate her. 
    
    Back in the cabin, Ash is attacked by Cheryl—who has escaped the cellar—and the reanimated Scott. Ash manages to 
    throw the Naturan Demanto into the fireplace. As the book burns, Scott and Cheryl gruesomely decompose, leaving the 
    disgusted Ash covered in their blood and entrails. As day breaks, Ash stumbles outside. While still standing outside, 
    the entity attacks him from behind, causing Ash to turn and scream. '''

    assert_prediction(the_evil_dead, y_test, get_file_path('imdb_dataset.csv'))

    y_test = ["Sci-Fi"]
    wall_e = '''In the 29th century, Earth has been abandoned and covered in garbage, its population having been 
    evacuated by the megacorporation Buy-N-Large (BnL) on giant starliners seven centuries earlier after decades of 
    mass consumerism facilitated by BnL. BnL has left behind WALL-E robot trash compactors to clean up; however, 
    all have since stopped functioning, except one unit who has gained sentience and is scavenging parts from other 
    units to remain active. One day, WALL-E discovers a healthy seedling, which he returns to his trailer home. 
    Later, an unmanned spaceship lands and deploys an EVE probe to scan the planet for plant life. WALL-E is 
    infatuated with EVE, who is initially hostile but gradually befriends him. When WALL-E brings EVE to his trailer 
    and shows her the plant, however, she suddenly takes the plant inside her and goes into standby mode. WALL-E, 
    confused, unsuccessfully tries to reactivate her. The ship then returns to collect EVE, and with WALL-E clinging 
    on, returns to its mothership, the starliner Axiom. 

    The Axiom's passengers have become obese and feeble due to microgravity and reliance on an automated lifestyle, 
    including the ship's current captain, McCrea, who leaves the ship under the control of the robotic autopilot, 
    AUTO. EVE is taken to the bridge, with WALL-E tagging along. McCrea is unprepared for a positive probe response, 
    but learns that placing EVE's plant in the ship's Holo-Detector for verification will trigger a hyperjump back to 
    Earth so humanity can recolonize it. The plant proves to be missing from EVE's storage compartment, though, 
    and she blames WALL-E for its disappearance. 
    
    With the plant missing, EVE is deemed faulty and taken to Diagnostics. WALL-E misinterprets the procedure as torture, 
    and in intervening accidentally frees a group of malfunctioning robots and causes both EVE and himself to be 
    designated as rogue robots. Frustrated, EVE takes WALL-E to an escape pod to send him home to retrieve the plant, 
    but they are interrupted when first mate robot GO-4 arrives with the plant, having stolen it from EVE on AUTO's 
    orders. GO-4 places the plant in an escape pod and sets it to self-destruct, but WALL-E enters just before it is 
    jettisoned. WALL-E escapes, saving the plant, and he and EVE reconcile and celebrate with a dance in space around the 
    Axiom. 
    
    EVE brings the plant back to McCrea, who watches EVE's recordings of Earth and concludes that they have to return to 
    clean up. However, AUTO refuses, revealing his own secret no-return directive A113, issued to BnL autopilots after 
    the corporation concluded in 2110 that the planet could not be saved. AUTO mutinies, electrocuting WALL-E and 
    shutting EVE down, and throwing them both down the garbage chute before confining the captain to his quarters. EVE 
    automatically reboots herself and helps WALL-E bring the plant to the ship's Holo-Detector chamber; AUTO tries to 
    close the chamber, crushing WALL-E as he struggles to keep it open. Seeing WALL-E's sacrifice, McCrea successfully 
    deactivates AUTO while EVE inserts the plant to activate the hyperjump. 
    
    Arriving back on Earth, EVE repairs WALL-E, but finds that his memory has been reset and his personality is gone. 
    Heartbroken, EVE gives WALL-E a farewell kiss, which sparks his memory and restores his original personality. WALL-E 
    and EVE reunite as the humans and robots of the Axiom begin to restore Earth and its environment. During the credits, 
    scenes of the humans and robots learning to farm, fish, and build are shown in various art styles, 
    with the implication being that Earth is turned into a paradise over several generations. '''

    assert_prediction(wall_e, y_test, get_file_path('imdb_dataset.csv'))


def assert_prediction(text, y_true, filename):
    data_loader = CsvLoader(filename, ["ID", "Text", "Category"])
    x_train, x_d, y_train, y_d, vec2ids = data_loader.get_vectors(pytest.x_vec, pytest.y_vec, 0)
    predictor = core.Predictor()
    predictor.fit(x_train, y_train)

    x_test = pytest.x_vec.transform([text])
    predicted = predictor.predict(x_test)
    y_predicted = pytest.y_vec.inverse_transform(predicted)
    matches = core.find_match(vec2ids, x_test)
    print("Closest match: {0}".format(matches))

    assert (check_equal(y_predicted, y_true) is True)
    assert (len(matches) > 0)


def check_equal(l1, l2):
    return len(l1) == len(l2) and sorted(l1) == sorted(l2)
