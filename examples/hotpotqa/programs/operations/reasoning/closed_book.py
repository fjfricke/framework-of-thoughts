from types import NoneType
from examples.hotpotqa.programs.operations.utils import find_dependencies, replace_dependencies

from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs


def prompter_hotpotqa(question: str, dependency_answers: list[str]) -> str:
    context = """
Please answer the question by thinking step-by-step. End with \"So the answer is: <answer>.\"
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: The South Korean television series in which Ryu Hye−young played Bo−ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is: 20.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: The movie Vertical Limit starred actors including Chiris O'Donnell, Robin Tunney, Scott Glenn, etc. The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the actor who stars in Vertical Limit and played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the answer is: Scott Glenn.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is: 15,140.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is: Raoul Walsh.
Q: In what country was Lost Gravity manufactured?
A: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is: Germany.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: The debut album of the band "Hot Hot Heat" was "Make Up the Breakdown". The debut album of the band "The Operation M.D." was "We Have an Emergency". So the answer is: The Operation M.D..
Q: Was Lonny (magazine) was founded in 2009?
A: Lonny (magazine) was founded in 2009. So the answer is: yes.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: The Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" is David Hicks. David Hicks received his para−military training in Afghanistan. So the answer is: Afghanistan.
Q: Does The Border Surrender or Unsane have more members?
A: The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin. That is, it has 4 members. Unsane has following members: Chris Spencer, Cooper, and Jon Syverson. That is, it has 3 members. Thus, The Border Surrender has more members. So the answer is: The Border Surrender.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: James Paris Lee is best known for investing the Lee−Metford rifle and Lee–Enfield series of rifles. Lee–Enfield is often referred to by the acronym of SMLE. So the answer is: SMLE.
Q: Was Lonny (magazine) was founded in 2008?
A: Lonny (magazine) was founded in 2009. So the answer is: no.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: The father of Edburga of Minster−in−Thanet is King Centwine. Centwine died after 685. So the answer is: after 685.
Q: Were Lonny and Allure both founded in the 1990s?
A: Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s. So the answer is: no.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: The actor that stars as Joe Proctor on the series "Power" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Turtle Assante's last name is Assante. So the answer is: Assante.
Q: When was Jan de Bont born?
A: Jan de Bont was born on 22 October 1943. So the answer is: 22 October 1943.
Q: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
A: Nobody Loves You was written by John Lennon and released on the album Walls and Bridges. The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. So the answer is: Walls and Bridges.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: The singer of "A Girl Like Me" is Rihanna. In the American Music Awards of 2012, Rihana won one award. So the answer is: one.
Q: Are both Bruce Chatwin and O. Henry writers？
A: Bruce Chatwin was an English travel writer, novelist, and journalist. O. Henry was an American writer. So both Bruce Chatwin and O. Henry are writers. So the answer is: yes.
Q: Which city is Lake Wales Medical Center located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. So the answer is: Polk County, Florida.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: Dadi Denis studied at the Maryland college Morgan State University. In 1890, the university's name was changed to honor Reverend Lyttleton Morgan. So the answer is: Reverend Lyttleton Morgan.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: William Orman Beerman was born in Manhattan, Kansas. Manhattan, Kansas is the county seat of Riley County. So the answer is: Riley County.
Q: 
"""
    dependencies_ids: list[int] = find_dependencies(question)
    if dependency_answers is None:
        dependency_answers = []
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return context + f"Q: {replaced_question}\nA: "

def prompter_musique(question: str, dependency_answers: list[str]) -> str:
    context = """
Please answer the question by thinking step-by-step. End with \"So the answer is: <answer>.\"
Q: When did the first large winter carnival take place in the city where CIMI−FM is licensed to broadcast?
A: CIMI−FM is licensed to broadcast in Quebec City. The first large winter carnival in Quebec City took place in 1894. So the answer is: 1894.
Q: When was Neville A. Stanton's employer founded?
A: The employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. So the answer is: 1862.
Q: What religion did the black community found?
A: The black community found African Methodist Episcopal Church. So the answer is: African Methodist Episcopal Church.
Q: What county is Hebron located in, in the same province the Heritage Places Protection Act applies to?
A: Heritage Places Protection Act applies to the jurisdiction of Prince Edward Island. Hebron, Prince Edward Island is located in the Prince County. So the answer is: Prince County.
Q: What weekly publication in the Connecticut city with the most Zagat rated restaurants is issued by university of America−Lite: How Imperial Academia Dismantled Our Culture's author?
A: The author of America−Lite: How Imperial Academia Dismantled Our Culture is David Gelernter. David Gelernter was educated at the Yale University. The city in Connecticut that has the highest number of Zagat−rated restaurants is New Haven. The weekly publication in New Haven that is issued by Yale University is Yale Herald. So the answer is: Yale Herald.
Q: What is the headquarters for the organization who sets the standards for ISO 21500?
A: The standards for ISO 21500 were set by International Organization for Standardization. The International Organization for Standardization has headquarters in Geneva. So the answer is: Geneva.
Q: What did the publisher of Banjo−Tooie rely primarily on for its support?
A: The publisher of Banjo−Tooie is Nintendo. Nintendo relied primarily for its support on first−party games. So the answer is: first−party games.
Q: The Collegian was owned by?
A: The Collegian was owned by Houston Baptist University. So the answer is: Houston Baptist University.
Q: In which county was the birthplace of the Smoke in tha City performer?
A: The performer of Smoke in tha City is MC Eiht. MC Eiht's birthplace is Compton. Compton is located in the county of Los Angeles County. So the answer is: Los Angeles County.
Q: What region of the state where Guy Shepherdson was born, contains SMA Negeri 68?
A: Guy Shepherdson was born in Jakarta. SMA Negeri 68 Jakarta is located in Central Jakarta. So the answer is: Central Jakarta.
Q: When did Britain withdraw from the country containing Hoora?
A: Hoora is in the country of Bahrain. Britain withdrew from Bahrain in 1971. So the answer is: 1971.
Q: Where does the Snake River start, in the state where Lima Mountain is located?
A: Lima Mountain is located in the state of Minnesota. The snake river in Minnesota starts in southern Aitkin County. So the answer is: southern Aitkin County.
Q: What shares a border with RiviÃ´lre−Verte in the province WRSU−FM broadcasts in?
A: WRSU−FM was licensed to broadcast to New Brunswick. RiviÃ´lre−Verte, New Brunswick shares border with Edmundston. So the answer is: Edmundston.
Q: When was the state of emergency declared in the country where the Senate is located?
A: The Senate is in the country of Kenya. The state of emergency was declared in Kenya on 20 October 1952. So the answer is: 20 October 1952.
Q: How long is the US border with the country that borders the state where Finding Dory takes place?
A: Finding Dory is supposed to take place in California. The country that shares a border with California is Mexico. The length of the us border with Mexico is 1,989 mi. So the answer is: 1,989 mi.
Q: What genre is the record label of the performer of So Long, See You Tomorrow associated with?
A: The performer of So Long, See You Tomorrow is Bombay Bicycle Club. The record label of Bombay Bicycle Club is Island Records. The genre of Island Records is jazz. So the answer is: jazz.
Q: When did the first large winter carnival happen in Olivier Robitaille's place of birth?
A: Olivier Robitaille was born in Quebec City. The first large winter carnival in Quebec City happened in the 1894. So the answer is: 1894.
Q: What is the genre of the record label of the band that performed on the Crush Tour?
A: The Crush Tour is performed by the band Bon Jovi. The record label of Bon Jovi is Island Records. The genre of Island Records is jazz. So the answer is: jazz.
Q: When was the first railway line constructed between Kotri and the city where Marie Adelaide Leprosy Centre is located?
A: Marie Adelaide Leprosy Centre is located in Karachi. The first railway line between Kotri and Karachi was constructed in April 1858. So the answer is: April 1858.
Q: In which state is Hertfordshire located?
A: Hertfordshire is located in the state East of England. So the answer is: East of England.
Q: Where is the crying stone found in the country in which Raphael Tuju holds citizenship?
A: Raphael Tuju is a citizen of Kenya. The crying stone in Kenya is found along the highway towards Kisumu. So the answer is: along the highway towards Kisumu.
Q: When did Britain withdraw from the country where the village of Wadyan is found?
A: Wadyan is in the country of Bahrain. Britain withdraw from Bahrain in 1971. So the answer is: 1971.
Q: How many countries in Pacific National University's continent are recognized by the organization that mediated the truce ending the Iran−Iraq war?
A: Pacific National University is located in Khabarovsk, Russia Khabarovsk. Russian is in the continent of Asia. The organization that mediated the truce which ended the Iran−Iraq War is the UN. The number of member states that UN recognises in Asia is 53. So the answer is: 53.
Q: 
"""
    dependencies_ids: list[int] = find_dependencies(question)
    if dependency_answers is None:
        dependency_answers = []
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return context + f"Q: {replaced_question}\nA: "

def parser(data: list[tuple[str, float]]) -> dict[str, any]:
    # Extract tokens and logprobs
    tokens = [token for token, _ in data]
    logprobs = [logprob for _, logprob in data]
    
    # Calculate the decomposition score (average of logprobs)
    decomposition_score = sum(logprobs) / len(logprobs)
    
    # Concatenate tokens into a single string
    full_text = "".join(tokens)
    
    # Extract the answer from the concatenated string and strip the last full stop if it exists
    answer = full_text.split(":")[-1].strip().rstrip(".")

    return {
        "answer": answer,
        "decomposition_score": decomposition_score
    }

class ClosedBookReasoning(LLMOperationWithLogprobs):
    def __init__(self, llm: AbstractLanguageModel, use_cache: bool = True, params: dict = None, name: str = "ClosedBookReasoning", prompter: str = prompter_hotpotqa):
        input_types = {"question": str, "dependency_answers": list[str] | NoneType}
        output_types = {"answer": str, "decomposition_score": float}
        super().__init__(llm, prompter, parser, use_cache, params, input_types, output_types, name)


if __name__ == "__main__":
    import asyncio
    from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs

    llm = OpenAIChat(model="gpt-4o-mini")
    operation = ClosedBookReasoning(llm)
    answer = asyncio.run(operation._execute(None, {"question": "What is the capital of #1?", "dependency_answers": ["Paris"]}))
    print(answer)