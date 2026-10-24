evaluation_dataset=[

{

    "question":"What is Diabetes?",
    "expected_answer":"Diabetes is a condition in which blood glucose (sugar) is too high because the body does not make enough insulin, cannot use insulin effectively, or both.",
    "reference_context":["Diabetes is a condition in which blood glucose (sugar) is too high because the body does not make enough insulin, cannot use insulin","Insulin helps glucose move from the bloodstream into cells for energy.","The right diagnosis and treatment depend on clinical assessment and testing."],
    "expected_tools": ["retriever_fun"]
},

{

    "question":"What are types of diabetes?",
    "expected_answer":"Type 1 diabetes occurs when the immune system destroys insulin-producing cells and insulin treatment is required. Type 2 diabetes involves insulin resistance and/or insufficient insulin production",
    "reference_context":["Type 1 diabetes occurs when the immune system destroys insulin-producing cells and insulin treatment is required.", "Type 2 diabetes involves insulin resistance and/or insufficient insulin production; it is influenced by many genetic, environmental, and lifestyle factors." ,"Diabetes is a condition in which blood glucose (sugar) is too high because the body does not make enough insulin, cannot use insulin"],
    "expected_tools": ["retriever_fun"]
}]