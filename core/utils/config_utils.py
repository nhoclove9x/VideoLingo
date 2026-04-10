from ruamel.yaml import YAML
import threading

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

DEFAULT_CONFIG_VALUES = {
    'merge_dub_into_video': True,
}

yaml = YAML()
yaml.preserve_quotes = True


def _persist_default_key(data, key, default_value):
    keys = key.split('.')
    current = data
    for part in keys[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value

    current[keys[-1]] = default_value
    return default_value

# -----------------------
# load & update config
# -----------------------

def load_key(key):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
                continue

            if key in DEFAULT_CONFIG_VALUES:
                default_value = _persist_default_key(data, key, DEFAULT_CONFIG_VALUES[key])
                with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                    yaml.dump(data, file)
                return default_value

            raise KeyError(f"Key '{k}' not found in configuration")
        return value

def update_key(key, new_value):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
