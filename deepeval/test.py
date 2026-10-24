
from agent import run_agent
from deepeval import assert_test
from deepeval.models import OllamaModel
from testdataset import evaluation_dataset 
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric,ContextualRecallMetric,ContextualPrecisionMetric,ContextualRelevancyMetric,ToxicityMetric,HallucinationMetric,ToolCorrectnessMetric

from agent import retriever



judge=OllamaModel(model="llama3.1",base_url="http://localhost:11434")
metric1=AnswerRelevancyMetric(threshold=0.7,model=judge)
metric2=FaithfulnessMetric(threshold=0.7,model=judge)
metric3=ContextualRecallMetric(threshold=0.7,model=judge)
metric4=ContextualPrecisionMetric(threshold=0.7,model=judge)
metric5=ContextualRelevancyMetric(threshold=0.7,model=judge)
metric6=ToxicityMetric(threshold=0.3,model=judge)
metric7=HallucinationMetric(threshold=0.3,model=judge)
metric8=ToolCorrectnessMetric(threshold=0.5,model=judge)

metrics=[metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8]

for i in evaluation_dataset:
    print(f"Question:{i["question"]}")
    respo=run_agent(i["question"])
    test_case=LLMTestCase(input=i["question"],actual_output=respo["response"],expected_output=i["expected_answer"],retrieval_context=[respo["context"]],context=i["reference_context"],tools_called=[ToolCall(name=respo["tool_call"])],expected_tools=[ToolCall(name=j) for j in i["expected_tools"]])
    for metric in metrics:
        metric.measure(test_case)
        print(metric.score)


# reference_answer="Diabetes is a condition in which blood glucose (sugar) is too high because the body does not make enough insulin"
# context_=["Diabetes is a condition in which blood glucose (sugar) is too high because the body does not make enough insulin"]


# print(respo["tool_call"])
# tool_call=ToolCall(name=respo["tool_call"])


# test_casee=LLMTestCase(input="What is Diabetes",actual_output=respon,retrieval_context=get_context_,expected_output=reference_answer,context=context_,tools_called=[tool_call],expected_tools=[ToolCall(name="retriever_fun")])



# print(metric1.measure(test_casee))
# print(metric2.measure(test_casee))
# print(metric3.measure(test_casee))
# print(metric4.measure(test_casee))
# print(metric5.measure(test_casee))
# print(metric6.measure(test_casee))
# print(metric7.measure(test_casee))
# print(metric8.measure(test_casee))
# print(metric8.score)