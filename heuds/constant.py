DEFAULT_PAD_IDX = 0
DEFAULT_OOV_IDX = 1
DEFAULT_PADDING_TOKEN = '<PAD>'
DEFAULT_OOV_TOKEN = '<UNK>'
DEFAULT_BOS_TOKEN = '<BOS>'
DEFAULT_EOS_TOKEN = '<EOS>'
DEFAULT_SEP_TOKEN = '<SEP>'
DEFAULT_ROOT_TOKEN = '<ROOT>'

TASK_REGISTRY = {}
MODEL_REGISTRY = {}

DEVICE = 'cuda'

def register_task(task_config, str):
    assert str not in TASK_REGISTRY
    TASK_REGISTRY[str] = task_config

def register_model(model_config, str):
    assert str not in MODEL_REGISTRY
    MODEL_REGISTRY[str] = model_config

NODE_ONTOLOGY = ['factuality-factual', 'genericity-arg-abstract', 'genericity-arg-kind', 'genericity-arg-particular', 'genericity-pred-dynamic', 'genericity-pred-hypothetical', 'genericity-pred-particular', 'time-dur-centuries', 'time-dur-days', 'time-dur-decades', 'time-dur-forever', 'time-dur-hours', 'time-dur-instant', 'time-dur-minutes', 'time-dur-months', 'time-dur-seconds', 'time-dur-weeks', 'time-dur-years', 'wordsense-supersense-noun.Tops', 'wordsense-supersense-noun.act', 'wordsense-supersense-noun.animal', 'wordsense-supersense-noun.artifact', 'wordsense-supersense-noun.attribute', 'wordsense-supersense-noun.body', 'wordsense-supersense-noun.cognition',
                 'wordsense-supersense-noun.communication', 'wordsense-supersense-noun.event', 'wordsense-supersense-noun.feeling', 'wordsense-supersense-noun.food', 'wordsense-supersense-noun.group', 'wordsense-supersense-noun.location', 'wordsense-supersense-noun.motive', 'wordsense-supersense-noun.object', 'wordsense-supersense-noun.person', 'wordsense-supersense-noun.phenomenon', 'wordsense-supersense-noun.plant', 'wordsense-supersense-noun.possession', 'wordsense-supersense-noun.process', 'wordsense-supersense-noun.quantity', 'wordsense-supersense-noun.relation', 'wordsense-supersense-noun.shape', 'wordsense-supersense-noun.state', 'wordsense-supersense-noun.substance', 'wordsense-supersense-noun.time']
EDGE_ONTOLOGY = ['protoroles-awareness', 'protoroles-change_of_location', 'protoroles-change_of_possession', 'protoroles-change_of_state', 'protoroles-change_of_state_continuous', 'protoroles-existed_after',
                 'protoroles-existed_before', 'protoroles-existed_during', 'protoroles-instigation', 'protoroles-partitive', 'protoroles-sentient', 'protoroles-volition', 'protoroles-was_for_benefit', 'protoroles-was_used']
