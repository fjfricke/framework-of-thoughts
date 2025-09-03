from examples.hotpotqa.programs.operations.utils import parse_tree_and_extract_logprobs
from llm_graph_optimizer.operations.helpers.exceptions import OperationFailed


def understanding_prompt_hotpotqa(question: str):
    examples = """Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: {"Jeremy Theobald and Christopher Nolan share what profession?": ["What is Jeremy Theobald's profession?", "What is Christopher Nolan's profession?"]}.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: {"How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?": ["In which South Korean television series Ryu Hye−young played Bo−ra?", "How many episodes were <1>?"]}.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: {"Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?": ["Vertical Limit stars which actor?", "Which actor played astronaut Alan Shepard in \"The Right Stuff\"?"]}.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: {"What was the 2014 population of the city where Lake Wales Medical Center is located?": ["Which city was Lake Wales Medical Center located in?", "What was the 2014 population of <1>?"]}.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: {"Who was born first? Jan de Bont or Raoul Walsh?": ["When was Jan de Bont born?", "When was Raoul Walsh born?"]}.
Q: In what country was Lost Gravity manufactured?
A: {"In what country was Lost Gravity manufactured?": ["Which company was Lost Gravity manufactured?", "Which country is <1> in?"]}.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: {"Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?": ["What is the debut album of the band Hot Hot Heat?", "What is the debut album of the band The Operation M.D.?"]}.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: {"In which country did this Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" receive para−military training?": ["Which Australian was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\"?", "In which country did <1> receive para−military training?"]}.
Q: Does The Border Surrender or Unsane have more members?
A: {"Does The Border Surrender or Unsane have more members?": ["How many members does The Border Surrender have?", "How many members does Unsane have?"]}.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: {"James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?": ["James Paris Lee is best known for investing the Lee−Metford rifle and which other rifle?", "<1> is often referred to by what acronymn?"]}.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: {"What year did Edburga of Minster−in−Thanet's father die?": ["Who is Edburga of Minster−in−Thanet's father?", "What year did <1> die?"]}.
Q: Were Lonny and Allure both founded in the 1990s?
A: {"Were Lonny and Allure both founded in the 1990s?": ["When was Lonny (magazine) founded?", "When was Allure founded?"]}.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: {"The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?": ["Which actor stars as Joe Proctor on the series \"Power\"?", "<1> played a character on \"Entourage\" that has what last name?"]}.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: {"How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?": ["Who is the singer of \"A Girl Like Me\"?", "How many awards did <1> win at the American Music Awards of 2012?"]}.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: {"Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?": ["Dadi Denis studied at which Maryland college?", "<1>'s name was changed in 1890 to honor what man?"]}.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: {"William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?": ["In which city in northeastern Kansas William Orman Beerman was born?", "<1> is the county seat of what county?"]}.
Q: """
    return examples + question + "```json"

def understanding_prompt_musique(question: str):
    examples = """Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.
Q: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast?
A: {"When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast?": ["Which city is CIMI−FM licensed to broadcast?", "When did the first large winter carnival take place in <1>?"]}.
Q: What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?
A: {"What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?": ["Which did Heritage Places Protection Act apply to the jurisdiction of?", "which country is Hebron, <1> located in?"]}.
Q: What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America−Lite: How Imperial Academia Dismantled Our Culture's author?
A: {"What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America−Lite: How Imperial Academia Dismantled Our Culture's author?": ["Which university was the author of America−Lite: How Imperial Academia Dismantled Our Culture educated at?", "What city in Connecticut has the highest number of Zagat−rated restaurants?", "What is the weekly publication in <2> that is issued by <1>?"], "Which university was the author of America−Lite: How Imperial Academia Dismantled Our Culture educated at?": ["Who is the author of America−Lite: How Imperial Academia Dismantled Our Culture?", "Which university was <1> educated at?"]}.
Q: What did the publisher of Banjo−Tooie rely primarily on for its support?
A: {"What did the publisher of Banjo−Tooie rely primarily on for its support?": ["What is the publisher of Banjo−Tooie?", "What did <1> rely primarily for its support on first−party games?"]}.
Q: In which county was the birthplace of the Smoke in tha City performer?
A: {"In which county was the birthplace of the Smoke in tha City performer?": ["What's the birthplace of the Smoke in tha City performer?", "Which country is <1> located in?"], "What's the birthplace of the Smoke in tha City performer?": ["Who is the performer of Smoke in tha City?", "Where was <1> born?"]}.
Q: What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?
A: {"What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?": ["Where was Guy Shepherdson born?", "what region of the state is SMA Negeri 68 <1> located in?"]}.
Q: When did Britain withdraw from the country containing Hoora?
A: {"When did Britain withdraw from the country containing Hoora?": ["Which country is Hoora in?", "When did Britain withdraw from <1>?"]}.
Q: How long is the US border with the country that borders the state where Finding Dory takes place?
A: {"How long is the US border with the country that borders the state where Finding Dory takes place?": ["Which country shares a border with the state where Finding Dory is supposed to take place?", "how long is the us border with <1>?"], "Which country shares a border with the state where Finding Dory is supposed to take place?": ["where is finding dory supposed to take place", "which country shares a border with <1>"]}.
Q: When did the first large winter carnival happen in Olivier Robitaille's place of birth?
A: {"When did the first large winter carnival happen in Olivier Robitaille's place of birth?": ["Where was Olivier Robitaille born?", "when did the first large winter carnival take place in <1>?"]}.
Q: When did Britain withdraw from the country where the village of Wadyan is found?
A: {"When did Britain withdraw from the country where the village of Wadyan is found?": ["Which country is Wadyan in ?", "When did Britain withdraw from <1>?"]}.
Q: How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran−Iraq war?
A: {"How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran−Iraq war?": ["What continent is the country of Pacific National University located in?", "Who mediated the truce which ended the Iran-Iraq War?", "the <2> recognises how many regions in <1>?"], "What continent is the country of Pacific National University located in?": ["which country is Pacific National University located in?", "What continent is <1> in?"]}.
Q: When was Eritrea annexed by the Horn of Africa country where, along with Somalia and the country where Bissidiro is located, Somali people live?
A: {"When was Eritrea annexed by the Horn of Africa country where, along with Somalia and the country where Bissidiro is located, Somali people live?": ["Along with Kenya, the country where Bissidiro is located and Somalia, in what Horn of Africa country do Somali people live?", "When was Eritrea annexed by <1>?"], "Along with Kenya, the country where Bissidiro is located and Somalia, in what Horn of Africa country do Somali people live?": ["Which country is Bissidiro located in?", "Along with Kenya, <1> and Somalia, in what Horn of Africa country do Somali people live?"]}.
Q: What was used to launch the probe of the country where Gao is located to the planet where Hephaestus Fossae is found?
A: {"What was used to launch the probe of the country where Gao is located to the planet where Hephaestus Fossae is found?": ["Where was Goa?", "Where is Hephaestus Fossae found?", "<1> 's mangalyaan was sent to the <2> by launching what?"]}.
Q: Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?
A: {"Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?": ["What country is along with Eisenhower's VP's country, recognized Gaddafi's government early on?", "Where is the lowest place in the <1>"], "What country is along with Eisenhower's VP's country, recognized Gaddafi's government early on?": ["Eisenhower's vice president was a president of what country?", "Along with the <1> , what major power recognized Gaddafi's government at an early date?"], "Eisenhower's vice president was a president of what country?": ["Who served as Eisenhower's vice president?", "<1> was a president of what country?"]}.
Q: When did the capital of Virginia moved from John Nicholas's birth city to Charles Oakley's alma mater's city?
A: {"When did the capital of Virginia moved from John Nicholas's birth city to Charles Oakley's alma mater's city?": ["Which city was Charles Oakley's university located in?", "Where was John Nicholas born?", "When did the capital of virginia moved from <2> to <1>"], "Which city was Charles Oakley's university located in?": ["Which university was Charles Oakley educated at?", "Which city was <1> located in?"]}.
Q: """

    return examples + question + "```json"

def understanding_parser(output: list) -> dict:
    hqdt = parse_tree_and_extract_logprobs(list(map(lambda x: x[0], output)), list(map(lambda x: x[1], output)))
    if hqdt is None:
        raise OperationFailed("Failed to parse HQDT")
    return {"hqdt": parse_tree_and_extract_logprobs(list(map(lambda x: x[0], output)), list(map(lambda x: x[1], output)))}
