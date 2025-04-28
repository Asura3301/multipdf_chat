# Google Search Fallback Chain
class GoogleSearchFallbackChain: 
    def __init__(self, rag_chain, serper_search_tool, llm):
        self.rag_chain = rag_chain
        self.serper_search_tool = serper_search_tool 
        self.llm = llm

    def invoke(self, inputs):
        # First try RAG
        response = self.rag_chain.invoke(inputs)
        answer = response["answer"]

        # Check if the answer indicates insufficient data AND if serper tool is available
        if "INSUFFICIENT DATA" in answer and self.serper_search_tool:
            print("Vector search yielded insufficient data. Trying Google Search...")
            try:
                # Get query first
                query = inputs["input"]
                if "chat_history" in inputs and inputs["chat_history"]:
                    # Create a simple string prompt for query extraction
                    query_instruction = f"Extract the main search query from this question: {query}"
                    query = self.llm.invoke(query_instruction).content
                    print(f"Reformulated query for Google: {query}")

                # Execute Google search using the tool's function
                search_results = self.serper_search_tool.func(query)
                # print(f"Google Raw Results: {search_results[:200]}...") # Debug print first 200 chars

                # Check if Google returned a meaningful result
                if search_results and search_results.strip() and not search_results.startswith("No good Google Result was found"):
                    print("Google Search returned results.")
                    # Create new context with web results
                    web_context = f"Web search results for '{query}':\n\n{search_results}" # Use the string result directly

                    # Generate answer using web results with a direct string prompt
                    web_prompt_str = (
                        "You are a helpful assistant. Use ONLY the following web search results "
                        "to answer the question. Summarize the key points concisely. "
                        "If the results don't answer the question, say 'The web search did not provide a clear answer.'\n\n"
                        f"Web search results:\n{web_context}\n\n"
                        f"Question: {query}"
                    )

                    web_answer = self.llm.invoke(web_prompt_str).content
                    return {"answer": f"Based on web search: {web_answer} [Web Search - Google]"} # Indicate source

                else:
                    print("Google Search returned no useful results or an error string.")
                    # Return the original 'INSUFFICIENT DATA' response
                    return response

            except Exception as e:
                # Handle potential errors during the fallback process
                print(f"Error during Google Search fallback: {e}")
                # Return the original 'INSUFFICIENT DATA' response on error
                return response

        return response