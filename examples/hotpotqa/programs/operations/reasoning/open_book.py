from pathlib import Path
from types import NoneType
import bm25s
from examples.hotpotqa.programs.operations.utils import find_dependencies, replace_dependencies

from llm_graph_optimizer.graph_of_operations.graph_partitions import GraphPartitions
from llm_graph_optimizer.graph_of_operations.types import ReasoningState
from llm_graph_optimizer.language_models.abstract_language_model import AbstractLanguageModel
from llm_graph_optimizer.language_models.openai_chat import OpenAIChat
from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs

def get_retriever(save_dir: Path):
    return bm25s.BM25.load(save_dir, load_corpus=True, mmap=True)

def prompter_hotpotqa(question: str, dependency_answers: list[str], context: str) -> str:
    dependencies_ids: list[int] = find_dependencies(question)
    if dependency_answers is None:
        dependency_answers = []
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    context_with_examples = f"""
Please answer the question and explain why. Output no more than 5 words after "So the answer is". End with \"So the answer is: <answer>.\"

#1 Wikipedia Title: First (magazine)
Text: FiRST is a Singaporean movie magazine formerly published monthly, now running as a weekly newspaper insert.
#2 Wikipedia Title: Arthur's Magazine
Text: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book".
#3 Wikipedia Title: First for Women
Text: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies.
#4 Wikipedia Title: First Eleven (magazine)
Text: First Eleven is a British specialist magazine for parents of children at independent schools.
#5 Wikipedia Title: Earth First! (magazine)
Text: Earth First!, the radical environmental journal, is the official publication of the Earth First! movement. First published as a newsletter in 1980, it has existed alongside the movement as a way to spread commonly held beliefs in "Earth First!" culture, such as biocentrism, deep ecology, and direct action. The magazine is also commonly known as the "Earth First! Journal".
Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine was started in 1844. First for Women was started in 1989. So Arthur's Magazine was started first. So the answer is: Arthur's Magazine.

#1 Wikipedia Title: The Oberoi Group
Text: The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
#2 Wikipedia Title: The Body Has a Head
Text: The Body Has a Head is an album by King Missile frontman John S. Hall, released exclusively in Germany in 1996. Though billed as a Hall "solo album," the collection features considerable input from multi-instrumentalists Sasha Forte, Bradford Reed, and Jane Scarpantoni, all of whom would become members of the next incarnation of King Missile ("King Missile III") and contribute to that group's "debut" album, 1998's "Failure."
#3 Wikipedia Title: Oberoi family
Text: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
#4 Wikipedia Title: Has-a
Text: In database design, object-oriented programming and design (see object oriented program architecture), has-a (has_a or has a) is a composition relationship where one object (often called the constituted object, or part/constituent/member object) "belongs to" (is part or member of) another object (called the composite type), and behaves according to the rules of ownership. In simple words, has-a relationship in an object is called a member field of an object. Multiple has-a relationships will combine to form a possessive hierarchy.
#5 Wikipedia Title: Oberoi Realty
Text: Oberoi Realty is a real estate developer based in Mumbai, Maharashtra. It is led by Mr. Vikas Oberoi, CMD. The company has developed over 39 projects at locations across Mumbai. Its main interest is in Residential, Office Space, Retail, Hospitality and Social Infrastructure properties in Mumbai.
Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: The Oberoi family is part of a hotel company The Oberoi Group. The Oberoi Group has a head office in Delhi. So the answer is: Delhi.

#1 Wikipedia Title: 2014 Liqui Moly Bathurst 12 Hour
Text: The 2014 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 9 February 2014, was the twelfth running of the Bathurst 12 Hour.
#2 Wikipedia Title: 2015 Liqui Moly Bathurst 12 Hour
Text: The 2015 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 8 February 2015, was the thirteenth running of the Bathurst 12 Hour.
#3 Wikipedia Title: 2013 Liqui Moly Bathurst 12 Hour
Text: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour. The race also incorporated the opening round of the 2013 Australian GT Championship. The Australian GT Championship was to compete as the first hour only and cars were permitted to enter for only that hour or to cross-enter for both the first hour and continue for the endurance race.
#4 Wikipedia Title: Mount Panorama Circuit
Text: Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. It is situated on a hill with the dual official names of Mount Panorama and Wahluu and is best known as the home of the Bathurst 1000 motor race held each October, and the Bathurst 12 Hour event held each February. The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.
#5 Wikipedia Title: List of Mount Panorama races
Text: This is a list of significant car races that have been held at the Mount Panorama Circuit near Bathurst, New South Wales, Australia. As Australia's most famous motor racing circuit, Mount Panorama has had a significant influence on the history and industry of Australian motor racing.
Q: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
A: The 2013 Liqui Moly Bathurst 12 Hour was staged at the Mount Panorama Circuit. Mount Panorama Circuit is 6.213 km long. So the answer is: 6.213 km long.
{context}
Q: {replaced_question}
A: """
    dependencies_ids: list[int] = find_dependencies(question)
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return context_with_examples + f"Q: {replaced_question}\nA: "

def prompter_musique(question: str, dependency_answers: list[str], context: str) -> str:
    dependencies_ids: list[int] = find_dependencies(question)
    if dependency_answers is None:
        dependency_answers = []
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    context_with_examples = f"""
Please answer the question and explain why. Output no more than 5 words after "So the answer is". End with \"So the answer is: <answer>.\"

#1 Wikipedia Title: Wadyan
Text: Wadyan (Arabic: واديان) is a village in the island of Sitra, Bahrain. A branch of the National Bank of Bahrain and the Sitra police station are located in Wadyan.
#2 Wikipedia Title: Bahrain
Text: According to a January 2006 report by the United Nations Economic and Social Commission for Western Asia, Bahrain has the fastest-growing economy in the Arab world. Bahrain also has the freest economy in the Middle East and is twelfth-freest overall in the world based on the 2011 Index of Economic Freedom published by the Heritage Foundation/"Wall Street Journal".
#3 Wikipedia Title: The Benefit Company
Text: The Benefit Company (TBC) is the local switch in the Kingdom of Bahrain handling ATM and POS transactions among other services. Established in 1997 with a special license from the Central Bank of Bahrain as "Provider of Ancillary Services to the Financial Sector", it is the only financial network of its kind in the country.
#4 Wikipedia Title: British Empire
Text: While the Suez Crisis caused British power in the Middle East to weaken, it did not collapse. Britain again deployed its armed forces to the region, intervening in Oman (1957), Jordan (1958) and Kuwait (1961), though on these occasions with American approval, as the new Prime Minister Harold Macmillan's foreign policy was to remain firmly aligned with the United States. Britain maintained a military presence in the Middle East for another decade. In January 1968, a few weeks after the devaluation of the pound, Prime Minister Harold Wilson and his Defence Secretary Denis Healey announced that British troops would be withdrawn from major military bases East of Suez, which included the ones in the Middle East, and primarily from Malaysia and Singapore. The British withdrew from Aden in 1967, Bahrain in 1971, and Maldives in 1976.
#5 Wikipedia Title: Gulf Air
Text: Gulf Air ( "Ṭayarān al-Khalīj") is the flag carrier of Bahrain. Headquartered in Muharraq, adjacent to Bahrain International Airport, the airline operates scheduled services to 50 destinations in 28 countries across Africa, Asia and Europe. Its main base is Bahrain International Airport. It was formerly a multinational airline owned by Bahrain, UAE, Oman, and Qatar.
Q: When did Britain withdraw from the country where the village of Wadyan is found?
A: Wadyan is in the country of Bahrain. Britain withdraw from Bahrain in 1971. So the answer is: 1971.

#1 Wikipedia Title: So Long, See You Tomorrow (album)
Text: So Long, See You Tomorrow is the fourth album by the London indie rock band Bombay Bicycle Club, released on 3 February 2014. The album is named after the novel of the same name by William Maxwell.
#2 Wikipedia Title: Hallelujah I Love Her So
Text: ``Hallelujah I Love Her So ''Single by Ray Charles from the album Ray Charles (or, Hallelujah I Love Her So) B - side`` What Would I Do Without You'' Released 1956 Format 7 ''45rpm Recorded 1956 Genre soul rhythm and blues Length 2: 35 Label Atlantic Songwriter (s) Ray Charles Producer (s) Jerry Wexler Ray Charles singles chronology ``A Fool for You'' (1955)`` Hallelujah I Love Her So ''(1956) ``Mary Ann'' (1956)`` A Fool for You ''(1955) ``Hallelujah I Love Her So'' (1956)`` Mary Ann ''(1956)
#3 Wikipedia Title: See You on the Other Side (Mercury Rev album)
Text: See You on the Other Side is the third studio album by American neo-psychedelia band Mercury Rev, released in 1995 by record label Beggars Banquet.
#4 Wikipedia Title: Flaws (album)
Text: Flaws is the second studio album by the British indie rock band Bombay Bicycle Club, released on 9 July 2010 by Island Records. Unlike the band's previous releases, the album is entirely acoustic music, consisting of versions of their own tracks as well as cover versions of other artists. The album was produced in part by the guitarist Jamie MacColl's father, Neil MacColl, with recording taking place in February 2009 at The Church in Crouch End, London. The band started work on the album after completing their first album, "I Had the Blues But I Shook Them Loose".
#5 Wikipedia Title: The Antidote (Ronny Jordan album)
Text: The Antidote is the debut album by English jazz guitarist Ronny Jordan, that was released by Island Records in 1992.
Q: What genre is the record label of the performer of So Long, See You Tomorrow associated with?
A: The performer of So Long, See You Tomorrow is Bombay Bicycle Club. The record label of Bombay Bicycle Club is Island Records. Island Records released album The Antidote of English jazz guitarist Ronny Jordan. Thus Island Records is associated with jazz. So the answer is: jazz.

#1 Wikipedia Title: Dance in the Country
Text: Dance in the Country (French: "Danse à la campagne") is an 1883 oil painting by French artist Pierre-Auguste Renoir. It is currently kept at the Musée d'Orsay in Paris.
#2 Wikipedia Title: Josip Broz Tito
Text: In 1968, Tito offered Czechoslovak leader Alexander Dubček to fly to Prague on three hours notice if Dubček needed help in facing down the Soviets. In April 1969, Tito removed generals Ivan Gošnjak and Rade Hamović in the aftermath of the invasion of Czechoslovakia due to the unpreparedness of the Yugoslav army to respond to a similar invasion of Yugoslavia.
#3 Wikipedia Title: 1939 German ultimatum to Lithuania
Text: The 1939 German ultimatum to Lithuania was an oral ultimatum which Joachim von Ribbentrop, Foreign Minister of Nazi Germany, presented to Juozas Urbšys, Foreign Minister of Lithuania on 20 March 1939. The Germans demanded that Lithuania give up the Klaipėda Region (also known as the Memel Territory) which had been detached from Germany after World War I, or the Wehrmacht would invade Lithuania. The Lithuanians had been expecting the demand after years of rising tension between Lithuania and Germany, increasing pro-Nazi propaganda in the region, and continued German expansion. It was issued just five days after the Nazi occupation of Czechoslovakia. 
#4 Wikipedia Title: Slavs
Text: The word "Slavs" was used in the national anthem of the Slovak Republic (1939–1945), Yugoslavia (1943–1992) and the Federal Republic of Yugoslavia (1992–2003), later Serbia and Montenegro (2003–2006).
#5 Wikipedia Title: United States Army
Text: Currently, the army is divided into the Regular Army, the Army Reserve, and the Army National Guard. The army is also divided into major branches such as Air Defense Artillery, Infantry, Aviation, Signal Corps, Corps of Engineers, and Armor. Before 1903 members of the National Guard were considered state soldiers unless federalized (i.e., activated) by the President. Since the Militia Act of 1903 all National Guard soldiers have held dual status: as National Guardsmen under the authority of the governor of their state or territory and, when activated, as a reserve of the U.S. Army under the authority of the President.
Q: A country's military branch, the equivalent of which in the US contains the Air Defense Artillery, was unprepared for the invasion of the country occupied by the Nazi's. When was the word \"Slavs\" used in the national anthem of the unprepared country?
A: The Air Defense Artillery is a branch of the amry in the US. Nazi occupied Czechoslovakia in 1939. The army of Yugoslavia was unprepared for the invasion of Czechoslovakia. The word "Slavs" was used in the national anthem of Yugoslavia from 1943 to 1992. So the answer is: 1943–1992.
{context}
Q: {replaced_question}
A: """
    dependencies_ids: list[int] = find_dependencies(question)
    replaced_question = replace_dependencies(question, {id: answer for id, answer in zip(dependencies_ids, dependency_answers)})
    return context_with_examples + f"Q: {replaced_question}\nA: "


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

class OpenBookReasoning(LLMOperationWithLogprobs):
    def __init__(self, llm: AbstractLanguageModel, retriever: bm25s.BM25, use_cache: bool = True, k: int = 5, params: dict = None, name: str = "OpenBookReasoning", prompter = prompter_hotpotqa):
        input_types = {"question": str, "dependency_answers": list[str] | NoneType}
        output_types = {"answer": str, "decomposition_score": float}
        super().__init__(llm, prompter, parser, use_cache, params, input_types, output_types, name)
        self.retriever = retriever
        self.k = k

    async def retrieve_context(self, question: str) -> str:

        def parse_bm25_documents(bm25_result: bm25s.Results) -> str:
            context_str = ""
            documents = bm25_result.documents[0]
            for i, doc in enumerate(documents):
                context_str += f"\n#{i+1} Wikipedia Title: "
                context_str += doc["title"]
                context_str += "\nText: "
                context_str += "".join(doc["text"])
            return context_str
        
        question_tokenized = bm25s.tokenize(question, stopwords="en")
        context_raw = self.retriever.retrieve(question_tokenized, k=self.k, show_progress=False)
        context = parse_bm25_documents(context_raw)
        return context
    
    async def _execute(self, partitions: GraphPartitions, input_reasoning_states: ReasoningState) -> ReasoningState:
        input_reasoning_states["context"] = await self.retrieve_context(input_reasoning_states["question"])
        return await super()._execute(partitions, input_reasoning_states)


if __name__ == "__main__":
    import asyncio
    import os
    from llm_graph_optimizer.operations.llm_operations.llm_operation_with_logprobs import LLMOperationWithLogprobs

    llm = OpenAIChat(model="gpt-4o-mini")
    operation = OpenBookReasoning(llm, retriever=get_retriever(Path(os.getcwd()) / "examples" / "hotpotqa" / "dataset" / "HotpotQA" / "wikipedia_index_bm25"))
    answer = asyncio.run(operation._execute(None, {"question": "What is the capital of #1?", "dependency_answers": ["Paris"]}))
    print(answer)