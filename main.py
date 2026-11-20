from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
Elon Reeve Musk [ˈiːlɒn ˈɹiːv ˈmʌsk] (* 28. Juni 1971 in Pretoria, Südafrika) ist ein südafrikanisch-kanadisch-US-amerikanischer Unternehmer. Er wurde als Gründer und technischer Leiter des PayPal-Vorgängers X.com und des Raumfahrtunternehmens SpaceX sowie als Leiter und Mitinhaber des Elektroautoherstellers Tesla bekannt. Darüber hinaus gründete er weitere Unternehmen und hält seit 2022 eine Mehrheitsbeteiligung an dem Mikrobloggingdienst X (vormals Twitter).

Mit einem geschätzten Vermögen von etwa 700 Milliarden US-Dollar ist Musk der reichste Mensch der Welt. Mit seiner finanziellen und medialen Macht beeinflusst er in erheblichem Ausmaß – im Sinne seiner wirtschaftlichen und ideologischen Agenda – den öffentlichen politischen Diskurs weltweit. Neben seinen Aktivitäten in den Vereinigten Staaten, wo er der Republikanischen Partei Hunderte Millionen Dollar Wahlkampfhilfe zukommen ließ, unterstützt er rechtspopulistische und rechtsextreme Parteien in Europa und Südamerika. Durch seine Beiträge auf X wurde er auch für das Verbreiten von Verschwörungstheorien und für Äußerungen bekannt, die unter anderem als wissenschaftlich unhaltbar, irreführend, Panikmache, antisemitisch sowie als transphob kritisiert wurden.
    """
    summary_template = """
    given the information {information} about a person I want you to create>
    1. A short summary
    2. two interesting facts about them
    """

    summary_promt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_promt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
