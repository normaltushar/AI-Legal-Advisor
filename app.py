import streamlit as st
import chromadb
import os
import pandas as pd
import uuid
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Indian Legal Advisor", layout="wide")

# Initialize LLM client
llm = ChatGroq(
    temperature=0,
    groq_api_key='',
    model_name="llama-3.1-8b-instant"
)

# Initialize ChromaDB client & collection
client = chromadb.PersistentClient('legal_vectorstore')
legal_collection = client.get_or_create_collection(name="indian_legal_knowledge")

# Legal knowledge base creation function
def create_legal_knowledge_base():
    legal_data = [
        # Constitutional Rights
        {
            "provision": "Right to Constitutional Remedies (Article 32)",
            "content": "Guarantees the right to move the Supreme Court for enforcement of Fundamental Rights. The Supreme Court can issue writs like habeas corpus, mandamus, prohibition, quo warranto and certiorari. This is considered the 'heart and soul' of the Constitution.",
            "category": "constitutional_rights"
        },
        {
            "provision": "Right to Equality (Articles 14-18)",
            "content": "Article 14 guarantees equality before law and equal protection of laws. Article 15 prohibits discrimination on grounds of religion, race, caste, sex, or place of birth. Article 16 ensures equality of opportunity in public employment. Article 17 abolishes untouchability. Article 18 abolishes titles except military or academic distinctions.",
            "category": "constitutional_rights"
        },
        {
            "provision": "Right to Freedom (Articles 19-22)",
            "content": "Article 19 protects six freedoms: speech and expression, assembly, association, movement, residence, and profession. Article 20 provides protection against arbitrary conviction. Article 21 guarantees right to life and personal liberty. Article 22 provides protection against arrest and detention in certain cases.",
            "category": "constitutional_rights"
        },
        {
            "provision": "Right against Exploitation (Articles 23-24)",
            "content": "Article 23 prohibits human trafficking and forced labor. Article 24 prohibits employment of children below 14 years in factories, mines, or other hazardous employment.",
            "category": "constitutional_rights"
        },
       
        # Criminal Law and Procedure
        {
            "provision": "Indian Penal Code - Basic Framework",
            "content": "The IPC (1860) is the primary criminal code covering all substantive aspects of criminal law. It defines crimes and prescribes punishments for them, categorizing offenses against the body, property, public tranquility, state, etc.",
            "category": "criminal_law"
        },
        {
            "provision": "Criminal Procedure Code - FIR (Section 154)",
            "content": "First Information Report (FIR) is the document prepared by police upon receiving information about a cognizable offense. Police are duty-bound to register FIR without conducting preliminary inquiry. Citizens can approach Superintendent of Police if police refuse to register FIR.",
            "category": "criminal_procedure"
        },
        {
            "provision": "Criminal Procedure Code - Bail (Sections 436-450)",
            "content": "Provisions governing bail in bailable and non-bailable offenses. In bailable offenses, bail is a matter of right. In non-bailable offenses, bail is discretionary. Courts consider factors like severity of offense, evidence strength, flight risk, and possibility of witness tampering.",
            "category": "criminal_procedure"
        },
        {
            "provision": "Criminal Procedure Code - Trial Procedure",
            "content": "Outlines procedures for criminal trials including summons trials, warrant trials, and summary trials. Establishes procedural safeguards like right to be defended, right to know the accusation, and right against self-incrimination.",
            "category": "criminal_procedure"
        },
        {
            "provision": "Indian Penal Code - Theft (Section 378)",
            "content": "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property is said to commit theft. Punishment under Section 379 can extend to 3 years imprisonment, fine, or both.",
            "category": "criminal_law"
        },
        {
            "provision": "Indian Penal Code - Cheating (Section 415)",
            "content": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person to deliver any property or consent to the retention of property, or intentionally induces the person to do or omit anything which he would not do or omit if he were not so deceived, is said to 'cheat'. Punishment under Section 417 can extend to 1 year imprisonment, fine, or both.",
            "category": "criminal_law"
        },
       
        # Civil Law and Procedure
        {
            "provision": "Civil Procedure Code - Overview",
            "content": "The Civil Procedure Code, 1908 governs procedures for filing and defending civil suits. It regulates jurisdiction of courts, execution of decrees, appeals, reviews, and references. It also covers issues like attachment before judgment and temporary injunctions.",
            "category": "civil_procedure"
        },
        {
            "provision": "Civil Procedure Code - Filing Suit (Order 4)",
            "content": "Procedure for instituting civil suits through proper plaint filing in court of jurisdiction with appropriate court fees. The plaint must contain material facts, legal basis of claim, relief sought, and proper verification.",
            "category": "civil_procedure"
        },
        {
            "provision": "Civil Procedure Code - Limitation (Section 3)",
            "content": "Every suit instituted, appeal preferred, and application made after the prescribed period shall be dismissed, although limitation has not been set up as a defense. The Limitation Act, 1963 prescribes specific time periods for different types of suits.",
            "category": "civil_procedure"
        },
        {
            "provision": "Civil Procedure Code - Summary Judgment (Order 13A)",
            "content": "Allows courts to decide a claim without a full trial when there is no real prospect of success for the defending party or where there is no compelling reason for the case to proceed to trial.",
            "category": "civil_procedure"
        },
       
        # Family Law
        {
            "provision": "Hindu Marriage Act, 1955",
            "content": "Governs marriage, separation, and divorce among Hindus. Section 5 outlines conditions for valid Hindu marriage. Section 13 provides grounds for divorce including adultery, cruelty, desertion, conversion, mental disorder, etc.",
            "category": "family_law"
        },
        {
            "provision": "Muslim Personal Law (Shariat) Application Act, 1937",
            "content": "Applies personal law to Muslims in matters relating to marriage, succession, inheritance, and charities. Under this, various forms of Muslim marriages and divorces (like talaq, khula, etc.) are recognized.",
            "category": "family_law"
        },
        {
            "provision": "Special Marriage Act, 1954",
            "content": "Provides for civil marriage irrespective of religion. Section 4 outlines conditions for marriage. Section 27 provides grounds for divorce. Marriage requires 30-day notice period and registration before Marriage Officer.",
            "category": "family_law"
        },
        {
            "provision": "Hindu Succession Act, 1956 (as amended)",
            "content": "Governs succession and inheritance among Hindus. After 2005 amendment, daughters have equal coparcenary rights as sons in ancestral property. Section 8 deals with succession for males, and Section 15 for females.",
            "category": "family_law"
        },
       
        # Contract and Commercial Law
        {
            "provision": "Indian Contract Act - Elements of Valid Contract (Section 10)",
            "content": "All agreements are contracts if made by free consent of parties competent to contract, for lawful consideration and with lawful object, and are not expressly declared void. Essential elements include offer, acceptance, consideration, capacity, free consent, lawful object.",
            "category": "contract_law"
        },
        {
            "provision": "Indian Contract Act - Breach of Contract (Section 73)",
            "content": "When a contract is broken, the party suffering from breach is entitled to compensation for loss or damage naturally arising from the breach. Compensation not given for remote or indirect loss or damage.",
            "category": "contract_law"
        },
        {
            "provision": "Sale of Goods Act, 1930",
            "content": "Governs contracts relating to sale of goods. Section 4 defines sale as transfer of ownership for price. Section 16 implies condition as to merchantable quality. Provides remedies for breach including damages, specific performance, rejection, etc.",
            "category": "commercial_law"
        },
        {
            "provision": "Companies Act, 2013",
            "content": "Comprehensive legislation governing company formation, management, dissolution. Classifies companies as private, public, OPC, etc. Establishes corporate governance norms and protects shareholder interests. NCLT/NCLAT established for corporate disputes.",
            "category": "corporate_law"
        },
       
        # Property Law
        {
            "provision": "Transfer of Property Act, 1882",
            "content": "Regulates transfer of property between living persons. Section 5 defines 'transfer of property'. Covers various modes of transfer like sale, mortgage, lease, gift, etc. Section 54 deals with sale, requiring registered instrument for immovable property worth over Rs. 100.",
            "category": "property_law"
        },
        {
            "provision": "Registration Act, 1908",
            "content": "Mandates registration of certain documents related to immovable property. Section 17 lists compulsorily registrable documents including sale deeds, gift deeds, mortgage deeds, lease deeds exceeding one year.",
            "category": "property_law"
        },
        {
            "provision": "Indian Easements Act, 1882",
            "content": "Governs easement rights (right to use another's property). Defines easements as right to enjoyment of another's land for specific purpose. Covers acquisition, extinction of easements. Examples include right of way, right to light.",
            "category": "property_law"
        },
       
        # Consumer Protection
        {
            "provision": "Consumer Protection Act, 2019",
            "content": "Establishes Consumer Disputes Redressal Commissions at district (up to Rs 1 crore), state (Rs 1-10 crore), and national levels (above Rs 10 crore). Covers product liability, unfair trade practices, and misleading advertisements. Provides for mediation, class action, simplified procedures.",
            "category": "consumer_law"
        },
       
        # Labor Law
        {
            "provision": "Industrial Disputes Act, 1947",
            "content": "Governs employer-employee relations in industrial establishments. Provides machinery for investigation and settlement of industrial disputes through works committees, conciliation officers, labor courts, tribunals. Regulates layoffs, retrenchment, closure.",
            "category": "labor_law"
        },
        {
            "provision": "Employees' Provident Funds Act, 1952",
            "content": "Provides for compulsory contributory provident fund for employees. Applicable to establishments with 20+ employees. Employer and employee both contribute 12% of basic wages. Administered by EPFO.",
            "category": "labor_law"
        },
       
        # Intellectual Property
        {
            "provision": "Patents Act, 1970",
            "content": "Governs patent protection in India. Patents granted for inventions that are novel, involve inventive step, and have industrial application. Patent term is 20 years. Section 3 lists non-patentable subject matter.",
            "category": "intellectual_property"
        },
        {
            "provision": "Copyright Act, 1957",
            "content": "Protects literary, dramatic, musical, artistic works, films, sound recordings. Copyright exists for lifetime of author plus 60 years. Provides exclusive rights to reproduce, publish, perform, translate, adapt, etc.",
            "category": "intellectual_property"
        },
       
        # Procedural Aspects
        {
            "provision": "Public Interest Litigation",
            "content": "Legal action initiated in court for protection of public interest. Can be filed by any public-spirited individual or organization for enforcement of constitutional rights of disadvantaged groups or matters of public importance. Relaxed rules of standing and procedure.",
            "category": "litigation"
        },
        {
            "provision": "Alternative Dispute Resolution",
            "content": "Includes arbitration, mediation, conciliation, negotiation. Arbitration governed by Arbitration and Conciliation Act, 1996. Provides for binding decisions outside court. Mediation centers established in many courts for amicable settlement.",
            "category": "litigation"
        },
        {
            "provision": "Legal Services Authorities Act, 1987",
            "content": "Establishes framework for free legal aid to weaker sections. NALSA at national level, SLSAs at state level, DLSAs at district level. Organizes Lok Adalats for amicable settlement of disputes.",
            "category": "legal_aid"
        },
       
        # Specialized Areas
        {
            "provision": "Information Technology Act, 2000",
            "content": "Legal framework for electronic governance and e-commerce. Recognizes electronic records and signatures. Defines cybercrimes like hacking, identity theft, cyber terrorism. Establishes adjudication process for cyber disputes.",
            "category": "cyber_law"
        },
        {
            "provision": "Prevention of Money Laundering Act, 2002",
            "content": "Prevents money-laundering and confiscation of property derived from it. Imposes obligation on banking companies, financial institutions to maintain records, verify identity of clients, report suspicious transactions.",
            "category": "financial_law"
        },
        {
            "provision": "Right to Information Act, 2005",
            "content": "Promotes transparency in government functioning. Citizens can request information from public authorities. Establishes Information Commissions for appeals. Exemptions under Section 8 include national security, privacy, etc.",
            "category": "administrative_law"
        }
    ]
    
    return legal_data

def setup_legal_knowledge():
    if legal_collection.count() == 0:
        legal_data = create_legal_knowledge_base()
        for item in legal_data:
            legal_collection.add(
                documents=[item["content"]],
                metadatas=[{"provision": item["provision"], "category": item["category"]}],
                ids=[str(uuid.uuid4())]
            )
        st.success("Legal knowledge base initialized")
    else:
        st.info("Using existing legal knowledge base")

setup_legal_knowledge()

# Define the prompt template for legal analysis (human-readable format)
legal_analysis_template = """
You are an experienced Indian legal advisor. Analyze the following case and provide clear, actionable advice in well-structured format:

**Case Details:**
{case_description}

**Relevant Legal Provisions:**
{legal_provisions}

Please provide your analysis with these clear sections:

1. **Case Summary**:
   - Briefly summarize the key facts
   - Identify the main legal issues

2. **Applicable Laws**:
   - List the most relevant laws/sections
   - Explain how each law applies to this case
   - Mention possible punishments/remedies

3. **Recommended Actions**:
   - Step-by-step advice on what to do next
   - Suggested legal procedures to follow
   - Any immediate precautions to take

4. **Additional Considerations**:
   - Potential challenges or complications
   - Estimated timelines if relevant
   - Suggestions for documentation/evidence

Use clear headings, bullet points, and maintain a professional yet accessible tone. Focus on practical guidance rather than theoretical discussion.
"""

legal_analysis_prompt = PromptTemplate(
    input_variables=["case_description", "legal_provisions"],
    template=legal_analysis_template
)

# Query relevant legal provisions from vector store
def get_relevant_legal_provisions(case_description, top_n=8):
    results = legal_collection.query(
        query_texts=[case_description],
        n_results=top_n
    )
    provisions = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        provisions.append(f"{metadata['provision']}: {doc}")

    return "\n\n".join(provisions)

# Analyze case using prompt + LLM
def analyze_legal_case(case_description):
    legal_provisions = get_relevant_legal_provisions(case_description)
    chain = legal_analysis_prompt | llm
    response = chain.invoke({
        "case_description": case_description,
        "legal_provisions": legal_provisions
    })
    return response.content

# Save case history as JSON file
def save_case_history(case_description, analysis):
    case_data = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_description": case_description,
        "analysis": analysis
    }

    if not os.path.exists("case_history"):
        os.makedirs("case_history")

    filename = f"case_history/case_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(case_data, f, indent=2)

    return filename

# Streamlit UI
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Indian Legal Advisor")

st.sidebar.title("Case Analysis")
case_description = st.text_area("Enter your legal situation or case details:", height=150)

if st.button("Analyze Case") and case_description.strip():
    with st.spinner("Analyzing your case... Please wait."):
        try:
            analysis = analyze_legal_case(case_description)
            save_case_history(case_description, analysis)
            st.session_state.history.append((case_description, analysis))

            st.subheader("Legal Analysis")
            st.markdown(analysis)  # Using markdown to preserve formatting
            
            st.success("Analysis completed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again with more detailed case information.")

st.sidebar.subheader("Case History")
if st.session_state.history:
    for i, (desc, analysis) in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Case {i+1}: {desc[:50]}..."):
            st.write(f"**Description:** {desc}")
            if st.button(f"Show Full Analysis {i+1}"):
                st.subheader(f"Analysis for Case {i+1}")
                st.markdown(analysis)
else:
    st.sidebar.info("No case history yet. Submit a case to begin.")
