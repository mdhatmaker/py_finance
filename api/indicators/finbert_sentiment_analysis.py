from transformers import BertForSequenceClassification, BertTokenizer
import torch

# https://medium.com/@priyatoshanand/handle-long-text-corpus-for-bert-model-3c85248214aa


tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Source - https://www.brookings.edu/2022/08/11/the-future-of-crypto-regulation-highlights-from-the-brookings-event/

txt = """
1. HIGHER LEVELS OF RETAIL PARTICIPATION IN CRYPTO THAN TRADITIONAL COMMODITY MARKETS POSE UNIQUE CHALLENGES FOR REGULATORS.

One in five Americans report having traded cryptocurrency, and polls suggest crypto trading is more common among younger adults, men, and racial minorities. This is quite different from other financial instruments regulated by the CFTC, Benham noted. “You’re going to have more vulnerable investors… It’s incumbent on us to educate, to inform, to disclose risks involved.”

Michael Piwowar, a former Securities and Exchange Commissioner and now executive director of the Milken Institute Center for Financial Markets, linked increased Congressional attention to growth in retail crypto: “If you got one in five households that have interacted with crypto… [members of Congress] are going to start hearing it from their constituents.” Legislation to regulate digital assets has been introduced by Senators Lummis and Gillibrand, Stabenow and Boozman, and Toomey, as well as Representative Gottheimer. The Treasury is actively negotiating bipartisan stablecoin legislation with House Financial Services Committee Chair Waters and Ranking Member McHenry. Benham said that stablecoins, digital currency meant to always be equal to one dollar, are more of a “payment mechanism” and thus should be regulated by prudential banking regulators.

Digital asset regulation may require addressing crypto exchanges and digital wallets. American University Law Professor Hilary Allen noted that the stablecoin legislation under discussion does not, saying, “That is a gaping hole… Almost every major stablecoin… is affiliated with an exchange that profits from trading in that stablecoin.” Mark Wetjen, a former CFTC commissioner and current head of policy and regulatory strategy for FTX (one of the largest crypto exchanges), agreed: “The exchanges are the gateways to the entire crypto space, and so oversight of them is probably most important.” He pushed back that there was no current regulation, noting the requirement for state level licenses, such as New York’s Bitlicense: “If you want to list derivatives on bitcoin, for example, you need a license… so it may not be as dire a situation.”

2. CRYPTO CHALLENGES TRADITIONAL REGULATORY DISTINCTION BETWEEN SECURITIES AND COMMODITIES.

Traditionally, the SEC regulates securities while the CFTC regulates commodities and derivatives. Whether crypto is a security or commodity remains unclear, as various subcomponents of the crypto ecosystem challenge existing regulatory divisions. For instance, the SEC recently argued  that nine different crypto tokens were securities in an insider trading case while a federal judge ruled that virtual currency like Bitcoin constitutes a commodity.

Benham called on Congress to provide clarity on which of the hundreds – if not thousands – of coins in existence are securities versus commodities: “Ultimately, we’d like to see law drawing lines.” Piwowar said the lack of clarity creates unwelcome delays as many crypto-related applications before the SEC are “not getting answers” on whether their products represent securities. The result is that some crypto firms are “going outside the United States” to locate their business. Allen cautioned, though, that Congressional action could also constitute an indication that the government supports crypto. She warned against letting crypto into the regulated sphere for fear of giving it “implicit guarantees.”

A solution to the regulatory turf battle could be merging the SEC and CFTC, which Piwowar endorsed, as have many others. Congress, however, has shown little appetite to do so given the different Congressional committee jurisdictions involved.

3. CFTC WILL RESTRUCTURE TO BETTER PROTECT CONSUMERS AND MORE EFFECTIVELY REGULATE MARKETS.
Benham announced several changes at the CFTC during the Brookings event. First, LabCFTC will become the Office of Technology Innovation, reporting directly to the Chairman’s office. Behnam justified this by stating, “We are past the incubator stage, and digital assets and decentralized financial technologies have outgrown their sandboxes.” Second, CFTC’s Office of Customer Education and Outreach will be realigned within the Office of Public Affairs, which Behnam said would “leverage resources and a broader understanding of the issues facing the general public towards addressing the most critical needs in the most vulnerable communities.” Restructuring within a regulator may appear a bureaucratic shuffle but can reflect changes in internal power, agency focus, and prioritization. Directly reporting to the chair increases an office’s authority and prestige.

4. IS CRYPTO A PASSING FAD (OR WORSE, A BUBBLE THAT THREATENS FINANCIAL MARKETS)?
Allen argued that crypto is “purposely less efficient and more complicated than a more centralized system,” and does not have any societal value. FTX’s Wetjen disagreed: “The difference here with blockchain as the underpinning means by which you can transfer value is that there are absolutely no gates.” Piwowar broadly agreed with Wetjen that “We’re going to have the new generation of Amazons and Googles come out of this stuff,” but cautioned that while he was at the SEC, “Nine out of ten [crypto applications] were outright fraud, and then out of the one out of ten, nine out of ten of those were probably fraud.” Since January 2021, over 46,000 people have collectively lost over $1 billion to scams involving crypto.

Everyone wants to avoid a repeat of the 2008 global financial crisis. To do so, regulators have focused on avoiding and mitigating “systemic risk” to the financial system. Asked if he sees a “clear and present danger to the existing economic system,” Benham said he did not, pointing out that crypto is not sufficiently interconnected to pose systemic risk. He noted the decrease in crypto values over the past several months did not cause ripples in the financial system or the broader economy. Piwowar turned the question of systemic risk back onto the actions of financial regulators asking: “What is systemic risk?  It’s the risk that a federal policymaker is going to bail out a bank, either directly or indirectly.” Allen agreed that bailing out crypto would be a mistake quipping: “If anything should be able to fail, it should be crypto, which isn’t… funding productive economic capacity.”

Allen also noted the similarity in arguments centered on American global competitiveness which promoted lax regulation for derivatives: “It’s almost identical to the rhetoric we saw around swaps in the 1990s.” Credit default swaps, like crypto now, faced loose regulation and ultimately helped fuel the subprime mortgage crisis. Behnam noted that one of 2008’s biggest lessons was the need for the CFTC to promote market transparency in the “OTC [over-the-counter] derivative space.” Crypto proponents point to the underlying technology as being inherently more transparent, while critics point to the lack of understanding of aspects of the market, such as what assets back stablecoins like Tether.

5. DOES CRYPTO INCREASE FINANCIAL INCLUSION?
Cryptocurrency proponents frequently cite financial inclusion as a major benefit linking the higher usage of youth and communities of color who have higher rates of being unbanked or underbanked by traditional finance. Allen cautioned against “predatory inclusion” arguing that, “Because there’s no productive capacity behind them, their value derives from finding someone else to buy them from you.” Wetjen’s responded, blending his experience serving as a CFTC Commissioner with his time in the crypto industry: “From my own experience… at the CFTC, there’s plenty of authority that’s already in place for the agency to… be pretty thoughtful and relatively prescriptive, even in terms of what actually should be disclosed to, particularly, retail investors, or users of a platform such as FTX.” He argued that the right policy is “giving people the opportunity to be involved and invest in the space that they like but making sure that it’s done with the right safeguards.”

"""


tokens = tokenizer.encode_plus(txt, add_special_tokens=False)

tokens.keys()

#output
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

type(tokens['input_ids'])

#output
list


start = 0
window_length = 512

total_len = len(input_ids)

loop = True

while loop:
    end = start + window_length
    if end >= total_len:
        loop = False
        end = total_len
    print(f"start = {start}")
    print(f"end = {end}")
    start = end

#output
start = 0
end = 512
start = 512
end = 1024
start = 1024
end = 1536
start = 1536
end = 1653


def chunk_text_to_window_size_and_predict_proba(input_ids, attention_mask, total_len):
    """
    This function splits the given input text into chunks of a specified window length,
    applies transformer model to each chunk and computes probabilities of each class for each chunk.
    The computed probabilities are then appended to a list.

    Args:
        input_ids (List[int]): List of token ids representing the input text.
        attention_mask (List[int]): List of attention masks corresponding to input_ids.
        total_len (int): Total length of the input_ids.

    Returns:
        proba_list (List[torch.Tensor]): List of probability tensors for each chunk.
    """
    proba_list = []

    start = 0
    window_length = 510

    loop = True

    while loop:
        end = start + window_length
        # If the end index exceeds total length, set the flag to False and adjust the end index
        if end >= total_len:
            loop = False
            end = total_len

        # 1 => Define the text chunk
        input_ids_chunk = input_ids[start: end]
        attention_mask_chunk = attention_mask[start: end]

        # 2 => Append [CLS] and [SEP]
        input_ids_chunk = [101] + input_ids_chunk + [102]
        attention_mask_chunk = [1] + attention_mask_chunk + [1]

        # 3 Convert regular python list to Pytorch Tensor
        input_dict = {
            'input_ids': torch.Tensor([input_ids_chunk]).long(),
            'attention_mask': torch.Tensor([attention_mask_chunk]).int()
        }

        outputs = model(**input_dict)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=-1)
        proba_list.append(probabilities)


proba_list = chunk_text_to_window_size_and_predict_proba(input_ids, attention_mask, total_len)


def get_mean_from_proba(proba_list):
    """
    This function computes the mean probabilities of class predictions over all the chunks.

    Args:
        proba_list (List[torch.Tensor]): List of probability tensors for each chunk.

    Returns:
        mean (torch.Tensor): Mean of the probabilities across all chunks.
    """

    # Ensures that gradients are not computed, saving memory
    with torch.no_grad():
        # Stack the list of tensors into a single tensor
        stacks = torch.stack(proba_list)

        # Resize the tensor to match the dimensions needed for mean computation
        stacks = stacks.resize(stacks.shape[0], stacks.shape[2])

        # Compute the mean along the zeroth dimension (i.e., the chunk dimension)
        mean = stacks.mean(dim=0)

    return mean


mean = get_mean_from_proba(proba_list)
# tensor([0.0767, 0.1188, 0.8045])

torch.argmax(mean).item()

# output
2

# { “0”: “positive”, “1”: “negative”, “2”: “neutral”}



