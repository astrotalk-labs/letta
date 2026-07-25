import asyncio
import difflib
import hashlib
import inspect
import io
import os
import random
import re
import sys
import uuid
from collections.abc import Callable, Coroutine
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from logging import Logger
from typing import Any, TypeVar, cast, get_args, get_origin, get_type_hints
from urllib.parse import urljoin, urlparse

import demjson3 as demjson
import tiktoken
from pathvalidate import sanitize_filename as pathvalidate_sanitize_filename

import letta
from letta.constants import (
    CLI_WARNING_PREFIX,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    ERROR_MESSAGE_PREFIX,
    LETTA_DIR,
    MAX_FILENAME_LENGTH,
    TOOL_CALL_ID_MAX_LEN,
)
from letta.helpers.json_helpers import json_dumps, json_loads

_F = TypeVar("_F", bound=Callable[..., Any])

DEBUG: bool = os.environ.get("LOG_LEVEL") == "DEBUG"


ADJECTIVE_BANK = [
    "beautiful",
    "gentle",
    "angry",
    "vivacious",
    "grumpy",
    "luxurious",
    "fierce",
    "delicate",
    "fluffy",
    "radiant",
    "elated",
    "magnificent",
    "sassy",
    "ecstatic",
    "lustrous",
    "gleaming",
    "sorrowful",
    "majestic",
    "proud",
    "dynamic",
    "energetic",
    "mysterious",
    "loyal",
    "brave",
    "decisive",
    "frosty",
    "cheerful",
    "adorable",
    "melancholy",
    "vibrant",
    "elegant",
    "gracious",
    "inquisitive",
    "opulent",
    "peaceful",
    "rebellious",
    "scintillating",
    "dazzling",
    "whimsical",
    "impeccable",
    "meticulous",
    "resilient",
    "charming",
    "vivacious",
    "creative",
    "intuitive",
    "compassionate",
    "innovative",
    "enthusiastic",
    "tremendous",
    "effervescent",
    "tenacious",
    "fearless",
    "sophisticated",
    "witty",
    "optimistic",
    "exquisite",
    "sincere",
    "generous",
    "kindhearted",
    "serene",
    "amiable",
    "adventurous",
    "bountiful",
    "courageous",
    "diligent",
    "exotic",
    "grateful",
    "harmonious",
    "imaginative",
    "jubilant",
    "keen",
    "luminous",
    "nurturing",
    "outgoing",
    "passionate",
    "quaint",
    "resourceful",
    "sturdy",
    "tactful",
    "unassuming",
    "versatile",
    "wondrous",
    "youthful",
    "zealous",
    "ardent",
    "benevolent",
    "capricious",
    "dedicated",
    "empathetic",
    "fabulous",
    "gregarious",
    "humble",
    "intriguing",
    "jovial",
    "kind",
    "lovable",
    "mindful",
    "noble",
    "original",
    "pleasant",
    "quixotic",
    "reliable",
    "spirited",
    "tranquil",
    "unique",
    "venerable",
    "warmhearted",
    "xenodochial",
    "yearning",
    "zesty",
    "amusing",
    "blissful",
    "calm",
    "daring",
    "enthusiastic",
    "faithful",
    "graceful",
    "honest",
    "incredible",
    "joyful",
    "kind",
    "lovely",
    "merry",
    "noble",
    "optimistic",
    "peaceful",
    "quirky",
    "respectful",
    "sweet",
    "trustworthy",
    "understanding",
    "vibrant",
    "witty",
    "xenial",
    "youthful",
    "zealous",
    "ambitious",
    "brilliant",
    "careful",
    "devoted",
    "energetic",
    "friendly",
    "glorious",
    "humorous",
    "intelligent",
    "jovial",
    "knowledgeable",
    "loyal",
    "modest",
    "nice",
    "obedient",
    "patient",
    "quiet",
    "resilient",
    "selfless",
    "tolerant",
    "unique",
    "versatile",
    "warm",
    "xerothermic",
    "yielding",
    "zestful",
    "amazing",
    "bold",
    "charming",
    "determined",
    "exciting",
    "funny",
    "happy",
    "imaginative",
    "jolly",
    "keen",
    "loving",
    "magnificent",
    "nifty",
    "outstanding",
    "polite",
    "quick",
    "reliable",
    "sincere",
    "thoughtful",
    "unusual",
    "valuable",
    "wonderful",
    "xenodochial",
    "zealful",
    "admirable",
    "bright",
    "clever",
    "dedicated",
    "extraordinary",
    "generous",
    "hardworking",
    "inspiring",
    "jubilant",
    "kindhearted",
    "lively",
    "miraculous",
    "neat",
    "openminded",
    "passionate",
    "remarkable",
    "stunning",
    "truthful",
    "upbeat",
    "vivacious",
    "welcoming",
    "yare",
    "zealous",
]

NOUN_BANK = [
    "lizard",
    "firefighter",
    "banana",
    "castle",
    "dolphin",
    "elephant",
    "forest",
    "giraffe",
    "harbor",
    "iceberg",
    "jewelry",
    "kangaroo",
    "library",
    "mountain",
    "notebook",
    "orchard",
    "penguin",
    "quilt",
    "rainbow",
    "squirrel",
    "teapot",
    "umbrella",
    "volcano",
    "waterfall",
    "xylophone",
    "yacht",
    "zebra",
    "apple",
    "butterfly",
    "caterpillar",
    "dragonfly",
    "elephant",
    "flamingo",
    "gorilla",
    "hippopotamus",
    "iguana",
    "jellyfish",
    "koala",
    "lemur",
    "mongoose",
    "nighthawk",
    "octopus",
    "panda",
    "quokka",
    "rhinoceros",
    "salamander",
    "tortoise",
    "unicorn",
    "vulture",
    "walrus",
    "xenopus",
    "yak",
    "zebu",
    "asteroid",
    "balloon",
    "compass",
    "dinosaur",
    "eagle",
    "firefly",
    "galaxy",
    "hedgehog",
    "island",
    "jaguar",
    "kettle",
    "lion",
    "mammoth",
    "nucleus",
    "owl",
    "pumpkin",
    "quasar",
    "reindeer",
    "snail",
    "tiger",
    "universe",
    "vampire",
    "wombat",
    "xerus",
    "yellowhammer",
    "zeppelin",
    "alligator",
    "buffalo",
    "cactus",
    "donkey",
    "emerald",
    "falcon",
    "gazelle",
    "hamster",
    "icicle",
    "jackal",
    "kitten",
    "leopard",
    "mushroom",
    "narwhal",
    "opossum",
    "peacock",
    "quail",
    "rabbit",
    "scorpion",
    "toucan",
    "urchin",
    "viper",
    "wolf",
    "xray",
    "yucca",
    "zebu",
    "acorn",
    "biscuit",
    "cupcake",
    "daisy",
    "eyeglasses",
    "frisbee",
    "goblin",
    "hamburger",
    "icicle",
    "jackfruit",
    "kaleidoscope",
    "lighthouse",
    "marshmallow",
    "nectarine",
    "obelisk",
    "pancake",
    "quicksand",
    "raspberry",
    "spinach",
    "truffle",
    "umbrella",
    "volleyball",
    "walnut",
    "xylophonist",
    "yogurt",
    "zucchini",
    "asterisk",
    "blackberry",
    "chimpanzee",
    "dumpling",
    "espresso",
    "fireplace",
    "gnome",
    "hedgehog",
    "illustration",
    "jackhammer",
    "kumquat",
    "lemongrass",
    "mandolin",
    "nugget",
    "ostrich",
    "parakeet",
    "quiche",
    "racquet",
    "seashell",
    "tadpole",
    "unicorn",
    "vaccination",
    "wolverine",
    "xenophobia",
    "yam",
    "zeppelin",
    "accordion",
    "broccoli",
    "carousel",
    "daffodil",
    "eggplant",
    "flamingo",
    "grapefruit",
    "harpsichord",
    "impression",
    "jackrabbit",
    "kitten",
    "llama",
    "mandarin",
    "nachos",
    "obelisk",
    "papaya",
    "quokka",
    "rooster",
    "sunflower",
    "turnip",
    "ukulele",
    "viper",
    "waffle",
    "xylograph",
    "yeti",
    "zephyr",
    "abacus",
    "blueberry",
    "crocodile",
    "dandelion",
    "echidna",
    "fig",
    "giraffe",
    "hamster",
    "iguana",
    "jackal",
    "kiwi",
    "lobster",
    "marmot",
    "noodle",
    "octopus",
    "platypus",
    "quail",
    "raccoon",
    "starfish",
    "tulip",
    "urchin",
    "vampire",
    "walrus",
    "xylophone",
    "yak",
    "zebra",
]


def smart_urljoin(base_url: str, relative_url: str) -> str:
    """urljoin is stupid and wants a trailing / at the end of the endpoint address, or it will chop the suffix off"""
    if not base_url.endswith("/"):
        base_url += "/"
    return urljoin(base_url, relative_url)


def get_tool_call_id() -> str:
    return str(uuid.uuid4())[:TOOL_CALL_ID_MAX_LEN]


def _matches_type(value: Any, hint: Any) -> bool:
    origin = get_origin(hint)
    if origin is list and isinstance(value, list):
        element_type = get_args(hint)[0] if get_args(hint) else None
        items = cast(list[object], value)
        return all(isinstance(v, element_type) for v in items) if element_type else True
    if origin:
        return isinstance(value, origin)
    return isinstance(value, hint)


def _check_arg_types(hints: dict[str, Any], named_args: dict[str, Any]) -> None:
    for arg_name, arg_value in named_args.items():
        hint = hints.get(arg_name)
        if hint and not _matches_type(arg_value, hint):
            raise ValueError(f"Argument {arg_name} does not match type {hint}; is {arg_value}")


def enforce_types(func: _F) -> _F:
    """Enforces that values passed in match the expected types.
    Technically will handle coroutines as well.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        hints = {k: v for k, v in get_type_hints(func).items() if k != "return"}
        arg_names = inspect.getfullargspec(func).args
        positional = dict(zip(arg_names[1:], args[1:]))  # skip 'self'
        _check_arg_types(hints, {**positional, **kwargs})
        return func(*args, **kwargs)

    return cast(_F, wrapper)


def create_random_username() -> str:
    """Generate a random username by combining an adjective and a noun."""
    adjective = random.choice(ADJECTIVE_BANK).capitalize()
    noun = random.choice(NOUN_BANK).capitalize()
    return adjective + noun


def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@contextmanager
def suppress_stdout():
    """Used to temporarily stop stdout (eg for the 'MockLLM' message)"""
    new_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield
    finally:
        sys.stdout = old_stdout


def count_tokens(s: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Falling back to cl100k base for token counting.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(s))


def printd(*args: Any, **kwargs: Any):
    if DEBUG:
        print(*args, **kwargs)


def united_diff(str1: str, str2: str) -> str:
    lines1 = str1.splitlines(True)
    lines2 = str2.splitlines(True)
    diff = difflib.unified_diff(lines1, lines2)
    return "".join(diff)


def parse_json(string: str) -> dict[str, Any]:
    """Parse JSON string into JSON with both json and demjson"""
    result: Any | None = None
    try:
        result = json_loads(string)
        if not isinstance(result, dict):
            raise ValueError(f"JSON from string input ({string}) is not a dictionary (type {type(result)}): {result}")
        return cast(dict[str, Any], result)
    except Exception as e:
        print(f"Error parsing json with json package, falling back to demjson: {e}")

    try:
        result = demjson.decode(string)
        if not isinstance(result, dict):
            raise ValueError(f"JSON from string input ({string}) is not a dictionary (type {type(result)}): {result}")
        return cast(dict[str, Any], result)
    except demjson.JSONDecodeError as e:
        print(f"Error parsing json with demjson package (fatal): {e}")
        raise e


def _coerce_function_response_to_str(value: Any, strict: bool) -> str:
    if value is None:
        return "None"  # backcompat
    if isinstance(value, dict):
        if strict:
            raise ValueError(value)
        try:
            return json_dumps(cast(dict[str, Any], value))
        except Exception:
            raise ValueError(value)
    if strict:
        raise ValueError(value)
    try:
        return str(value)
    except Exception:
        raise ValueError(value)


def validate_function_response(function_response_string: Any, return_char_limit: int, strict: bool = False, truncate: bool = True) -> str:
    """Check to make sure that a function used by Letta returned a valid response. Truncates to return_char_limit if necessary.

    Responses need to be strings (or None) that fall under a certain text count limit.
    """
    if not isinstance(function_response_string, str):
        function_response_string = _coerce_function_response_to_str(function_response_string, strict)

    if truncate and len(function_response_string) > return_char_limit:
        print(
            f"{CLI_WARNING_PREFIX}function return was over limit ({len(function_response_string)} > {return_char_limit}) and was truncated"
        )
        function_response_string = f"{function_response_string[:return_char_limit]}... [NOTE: function output was truncated since it exceeded the character limit ({len(function_response_string)} > {return_char_limit})]"

    return function_response_string


def list_human_files():
    """List all humans files"""
    defaults_dir = os.path.join(letta.__path__[0], "humans", "examples")
    user_dir = os.path.join(LETTA_DIR, "humans")

    letta_defaults = os.listdir(defaults_dir)
    letta_defaults = [os.path.join(defaults_dir, f) for f in letta_defaults if f.endswith(".txt")]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return letta_defaults + user_added


def list_persona_files():
    """List all personas files"""
    defaults_dir = os.path.join(letta.__path__[0], "personas", "examples")
    user_dir = os.path.join(LETTA_DIR, "personas")

    letta_defaults = os.listdir(defaults_dir)
    letta_defaults = [os.path.join(defaults_dir, f) for f in letta_defaults if f.endswith(".txt")]

    if os.path.exists(user_dir):
        user_added = os.listdir(user_dir)
        user_added = [os.path.join(user_dir, f) for f in user_added]
    else:
        user_added = []
    return letta_defaults + user_added


def get_human_text(name: str, enforce_limit: bool = True):
    for file_path in list_human_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            human_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(human_text) > CORE_MEMORY_HUMAN_CHAR_LIMIT:
                raise ValueError(f"Contents of {name}.txt is over the character limit ({len(human_text)} > {CORE_MEMORY_HUMAN_CHAR_LIMIT})")
            return human_text

    raise ValueError(f"Human {name}.txt not found")


def get_persona_text(name: str, enforce_limit: bool = True):
    for file_path in list_persona_files():
        file = os.path.basename(file_path)
        if f"{name}.txt" == file or name == file:
            persona_text = open(file_path, "r", encoding="utf-8").read().strip()
            if enforce_limit and len(persona_text) > CORE_MEMORY_PERSONA_CHAR_LIMIT:
                raise ValueError(
                    f"Contents of {name}.txt is over the character limit ({len(persona_text)} > {CORE_MEMORY_PERSONA_CHAR_LIMIT})"
                )
            return persona_text

    raise ValueError(f"Persona {name}.txt not found")


def create_uuid_from_string(val: str):
    """
    Generate consistent UUID from a string
    from: https://samos-it.com/posts/python-create-uuid-from-random-string-of-words.html
    """
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return uuid.UUID(hex=hex_string)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the given filename to prevent directory traversal, invalid characters,
    and reserved names while ensuring it fits within the maximum length allowed by the filesystem.

    Parameters:
        filename (str): The user-provided filename.

    Returns:
        str: A sanitized filename that is unique and safe for use.
    """
    # Extract the base filename to avoid directory components
    filename = os.path.basename(filename)

    # Split the base and extension
    base, ext = os.path.splitext(filename)

    # External sanitization library
    base = pathvalidate_sanitize_filename(base)

    # Cannot start with a period
    if base.startswith("."):
        raise ValueError(f"Invalid filename - derived file name {base} cannot start with '.'")

    # Truncate the base name to fit within the maximum allowed length
    max_base_length = MAX_FILENAME_LENGTH - len(ext) - 33  # 32 for UUID + 1 for `_`
    if len(base) > max_base_length:
        base = base[:max_base_length]

    # Append a unique UUID suffix for uniqueness
    unique_suffix = uuid.uuid4().hex[:4]
    sanitized_filename = f"{base}_{unique_suffix}{ext}"

    # Return the sanitized filename
    return sanitized_filename


def get_friendly_error_msg(function_name: str, exception_name: str, exception_message: str):
    from letta.constants import MAX_ERROR_MESSAGE_CHAR_LIMIT

    error_msg = f"{ERROR_MESSAGE_PREFIX} executing function {function_name}: {exception_name}: {exception_message}"
    if len(error_msg) > MAX_ERROR_MESSAGE_CHAR_LIMIT:
        error_msg = error_msg[:MAX_ERROR_MESSAGE_CHAR_LIMIT]
    return error_msg


def parse_stderr_error_msg(stderr_txt: str, last_n_lines: int = 3) -> tuple[str, str]:
    """
    Parses out from the last `last_n_line` of `stderr_txt` the Exception type and message.
    """
    index = -(last_n_lines + 1)
    pattern = r"(\w+(?:Error|Exception)): (.+)$"
    for line in stderr_txt.split("\n")[:index:-1]:
        if "Error" in line or "Exception" in line:
            match = re.search(pattern, line)
            if match:
                return match.group(1), match.group(2)
    return "", ""


def run_async_task(coro: Coroutine[Any, Any, Any]) -> Any:
    """
    Safely runs an asynchronous coroutine in a synchronous context.

    If an event loop is already running, it uses `asyncio.ensure_future`.
    Otherwise, it creates a new event loop and runs the coroutine.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        _ = asyncio.get_running_loop()
        return asyncio.ensure_future(coro)
    except RuntimeError:
        return asyncio.run(coro)


def log_telemetry(logger: Logger, event: str, **kwargs: Any):
    """
    Logs telemetry events with a timestamp.

    :param logger: A logger
    :param event: A string describing the event.
    :param kwargs: Additional key-value pairs for logging metadata.
    """
    from letta.settings import log_settings

    if log_settings.verbose_telemetry_logging:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f UTC")  # More readable timestamp
        extra_data = " | ".join(f"{key}={value}" for key, value in kwargs.items() if value is not None)
        logger.info(f"[{timestamp}] EVENT: {event} | {extra_data}")


def make_key(*args: Any, **kwargs: Any):
    return str((args, tuple(sorted(kwargs.items()))))


def safe_create_task(coro: Coroutine[Any, Any, Any], logger: Logger, label: str = "background task"):
    async def wrapper():
        try:
            await coro
        except Exception as e:
            logger.exception(f"{label} failed with {type(e).__name__}: {e}")

    return asyncio.create_task(wrapper())
