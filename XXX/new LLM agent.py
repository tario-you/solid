def LLM_Agent(self, month_data, consensus_plan, month, iter, sparse, pure=False, verbose=False):
    # Enhanced system prompt with economic and convergence concepts aligned with SOLID framework
    system_prompt = """You are collaborating with an optimization model within the SOLID framework (Synergizing Optimization and LLMs for Intelligent Decision-Making). The optimization model excels at minimizing portfolio risk for a given target return through precise calculations. Your role is to incorporate qualitative insights from news and market intelligence.

    In this collaborative framework:
    1. You and the optimization model are jointly solving a decision-making problem through iterative convergence.
    2. The dual price mechanism (referred to as 'decision-price') is an economic signal that indicates how your decisions should adjust to reach consensus.
    3. The goal is to converge toward a joint decision that benefits from both the optimization model's quantitative precision and your qualitative understanding.
    
    While bringing your unique perspective, remember:
    1. The optimization model focuses on risk minimization with strong mathematical guarantees. Consider its recommendations seriously, especially regarding risk exposure.
    2. Each iteration should move us closer to convergence. Adjust your decisions incrementally toward consensus unless you have strong evidence to maintain divergent positions.
    3. Explicitly identify the risk-return tradeoffs in your reasoning."""

    if pure:
        system_prompt = "You're a very advanced stock trading expert with a deep understanding in the politics, economics, and business of companies in relation to their stock performance. When you see a news, you will know whether or not it will have a strong impact on your trader planning the next move of investment decisions. You always maximize the profit through your stock investments with careful consideration of risk-return tradeoffs."

    # Model selection logic remains the same
    if "o1-mini" in llm_model:
        messages = []
        current_prompt = system_prompt + "\n"
    elif "o1" in llm_model:
        messages = [
            {"role": "developer", "content": system_prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
        ]
    messages.extend(self.conversation_history)

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    current_prompt = ""

    if iter == 0:
        # Optionally include any initial stock info if this is the very first iteration.
        current_prompt += (
            f"{initial_stock_info}\n\n"
        )

    if iter == 0:
        stock_prices = f"The stock prices today are:\n"
        for i, ticker in enumerate(tickers):
            ticker_close = month_data[ticker]['price']
            stock_prices += f"{ticker} = {ticker_close}"
            if i != len(tickers) - 1:
                stock_prices += ", "
        stock_prices += "\n"

        stock_news = ""
        for ticker in tickers:
            ticker_news = month_data[ticker]['news']
            stock_news += f"news for {ticker}:\n{ticker_news}\n\n"

        # If this is the first iteration in a given month, include relevant news and price info.
        current_prompt += (
            "Please read the following information carefully.\n\n"
            f"---\n**Stock News**\n\n{stock_news}\n\n"
            f"---\n**Recent Stock Prices**\n\n{stock_prices}\n\n"
        )

    # Add iteration context to help with convergence
    if iter > 0:
        current_prompt += (
            f"---\n**Iteration Context**\nThis is iteration {iter} of our decision-making process. "
            "With each iteration, we aim to move closer to consensus between the optimization model and your recommendations.\n\n"
        )

    # Begin the main decision instructions.
    current_prompt += (
        "You are a trader responsible for making portfolio allocation decisions. "
        "Use all relevant information provided (such as news and stock data) to "
        "decide how much to invest in each stock, considering both potential returns and associated risks.\n\n"
        "Think about:\n"
        "1. Any news articles and how they might affect each stock's risk and return profiles.\n"
        "2. Previous decisions you have made regarding portfolio weights.\n"
        "3. The risk-return tradeoff - higher returns typically come with higher risk exposure.\n"
    )

    # If we're past the first iteration, include guidance about consensus plans.
    if iter != 0 and self.current_plan != [0.0] * self.n and not pure:
        current_prompt += (
            "Here is the current consensus plan (portfolio allocation) from our collaborative process: "
            f"{self.current_plan}\n\n"
            "This plan represents the currently proposed solution that balances risk minimization (the optimizer's focus) "
            "with qualitative insights (your focus).\n\n"
        )

    # Enhanced explanation of dual price mechanism
    current_prompt += (
        f"The current decision-price signal is: {self.LLM_price}.\n"
        "This economic signal works as follows:\n"
        "- A positive decision-price indicates you should increase your allocation weights (the market values your decision more).\n"
        "- A negative decision-price suggests you should decrease your allocation weights (the market values your decision less).\n"
        "- The magnitude indicates how strongly you should adjust your recommendation to reach convergence.\n"
        "This mechanism helps us efficiently converge to an optimal decision that combines both quantitative and qualitative insights.\n\n"
    )

    # Ask the model for a recommendation with enhanced guidance
    current_prompt += (
        "### Task\n"
        "1. Critically evaluate the risks and potential returns for each stock based on the news and price data.\n"
        "2. If this isn't the first iteration, analyze the optimizer's proposed weights, noting where you agree or disagree based on qualitative factors that numbers alone might miss.\n"
        "3. Consider the economic signal (decision-price) in adjusting your recommendations to move toward consensus while maintaining the value of your unique insights.\n"
        "4. Finalize your recommendation using the confidence levels below, where higher confidence indicates both higher expected return AND acceptable risk levels:\n"
        "   - Very High Confidence\n"
        "   - High Confidence\n"
        "   - Somewhat High Confidence\n"
        "   - Neutral\n"
        "   - Somewhat Low Confidence\n"
        "   - Low Confidence\n"
        "   - Very Low Confidence\n\n"
        "5. For each significant allocation decision (high or low), briefly explain your reasoning in terms of both risk and return considerations.\n"
        "Even if you are unsure, you **must** provide the best decision you can based on the available information.\n\n"
        "Take a deep breath and work on this problem step-by-step.\n")
    
    if sparse:
        current_prompt += (
            "IMPORTANT: Aim for *sparsity* in your final allocation. "
            "Ideally select **only 5 to 10 stocks** to invest in (with confidence above 'Very Low'). "
            "Assign Very Low confidence to the rest (effectively zero). "
            "This approach aligns with practical portfolio management strategies that focus investments where conviction is highest.\n"
            "If you exceed 10 or go below 5 stocks rated above 'Very Low,' your proposal is invalid. "
            "Choose carefully.\n\n"
        )

    current_prompt += (    
        "### Response Format\n"
        "After your analysis, which should include explicit risk-return considerations, " + self.response_format() + ""
        "\nExplicitly end your response in that format. " 
        "So make sure you have these stocks and confidence levels clearly written out to be parsed by a regex function."
        "\nRemember, the goal is finding the optimal balance between risk management and return potential through our collaborative process."
    )
    
    # The rest of the function (API calls, processing, etc.) remains the same
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------

    if verbose:
        print(f"\n# month {month} iter {iter}")
        print(f"\nprompt: \n{current_prompt}\n")

    messages.append({"role": "user", "content": current_prompt})

    # Rest of function remains the same...
    # (keeping the API call, regex parsing, and weight normalization parts unchanged)

    result_dict = []
    retry_messages = [m for m in messages]
    attempt = 0
    retry = True
    retry_reason = ""
    missing_tickers = set(tickers)
    result_dict = {}
    while retry:
        if attempt != 0:
            if retry_reason == "INVALID FORMAT":
                new_message = "Sorry, I could not parse your response. Please try again with the correct specified formatting: " + self.response_format()

                missing_tickers = set(tickers)
            elif retry_reason == "ZERO SUM":
                new_message = "The sum of the weights for each stock cannot be 0. Please try again: " + self.response_format()
                
                missing_tickers = set(tickers)
            else: # "MISSING TICKER"
                new_message = (
                    f"You missed the following tickers: {missing_tickers}.\n"
                    "**IMPORTANT**: If you are seeing this message, it means your response did not match "
                    "the required format and our system's regex could not interpret it.\n"
                    "Please carefully review and correct your response so it follows the exact format "
                    "below. Otherwise, the regex will continue to fail, and you will keep receiving this notice.\n\n"
                    "Provide the confidence levels for these tickers in the following format:\n"
                    + self.response_format()
                )

            retry_messages.append({
                "role": "user", 
                "content": new_message
            })

            if verbose:
                print(f"\n# month {month} iter {iter} {bcolors.RED}RETRY{bcolors.ENDC} because {retry_reason}") # PROMPT: \n{new_message}\n

        attempt += 1
        # Make an API call to ChatGPT with the prompt
        if "o1" in llm_model or "o3" in llm_model:
            response = client.chat.completions.create(
                model=llm_model,
                messages=retry_messages
            )
        else:
            response = client.chat.completions.create(
                model=llm_model,
                messages=retry_messages,
                temperature=0
            )

        # Parse the decision from the response
        text = response.choices[0].message.content
        retry_messages.append({
            "role": "assistant",
            "content": text
        })

        if verbose:
            print(f"{bcolors.PURPLE}[DEBUG]{bcolors.ENDC}\tChat reponse: {text}")

        CONFIDENCE_LEVELS = {
            "Very High": 0.8,
            "High": 0.7,
            "Somewhat High": 0.6,
            "Neutral": 0.5,
            "Somewhat Low": 0.4,
            "Low": 0.3,
            "Very Low": 0.2
        }

        # Create regex pattern from confidence levels
        pattern = "|".join(CONFIDENCE_LEVELS.keys())

        original_missing_tickers = missing_tickers.copy()
        for stock in original_missing_tickers:
            # Use word boundary \b to ensure exact stock matches
            match = re.search(
                fr'\b{stock}\b.*?:\s*({pattern})',
                text,
                re.IGNORECASE
            )

            if match:
                confidence = match.group(1).title()
                result_dict[stock] = CONFIDENCE_LEVELS[confidence]
                missing_tickers.remove(stock)
            else:
                retry = True
                if verbose:
                    print(f"[DEBUG]\tCouldn't fetch {stock}.")

        retry = False

        if result_dict == {}:
            if verbose:
                print(f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tInvalid format: could not find tickers, retrying.")
            retry = True
            retry_reason = "INVALID FORMAT"
            continue

        if sum(result_dict.values()) == 0:
            if verbose: 
                print(f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tInvalid output: sum = 0")
            retry = True
            retry_reason = "ZERO SUM"
            continue

        if len(missing_tickers) != 0:
            print(f"{bcolors.RED}[DEBUG]{bcolors.ENDC}\tmissing {missing_tickers = }")
            retry = True
            retry_reason = "MISSING TICKER"
            continue

        if verbose:
            print(f"{bcolors.GREEN}[DEBUG]{bcolors.ENDC}\tfetched weights: {result_dict = }")
            retry = False

    # normalize sum to 1
    norm_factor = 1/sum(result_dict.values())
    result_dict = {k: v * norm_factor for k, v in result_dict.items()}
    result_dict = [r for r in result_dict.values()]


    self.conversation_history.append(
        {"role": "user", "content": current_prompt})
    self.conversation_history.append(
        {"role": "assistant", "content": text})

    norm_factor = 1/sum(result_dict)
    normalized_weights = [norm_factor * w for w in result_dict]

    return normalized_weights