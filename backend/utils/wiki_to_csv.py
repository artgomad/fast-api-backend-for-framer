import wikipedia
import re
from typing import Set
# from transformers import GPT2TokenizerFast
import tiktoken
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


def filter_olympic_2020_titles(titles):
    """
    Get the titles which are related to Olympic games hosted in 2020, given a list of titles
    """
    titles = [
        title for title in titles if '2020' in title and 'olympi' in title.lower()]
    print("Titles")
    print(titles)

    return titles


def get_wiki_page(title):
    """
    Get the wikipedia page given a title
    """
    print("Page")
    print(title)
    try:
        return wikipedia.page(title)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0])
    except wikipedia.exceptions.PageError as e:
        return None


def recursively_find_all_pages(titles, titles_so_far=set()):
    """
    Recursively find all the pages that are linked to the Wikipedia titles in the list
    """

    print("Starting to look for pages")
    all_pages = []

    titles = list(set(titles) - titles_so_far)
    titles = filter_olympic_2020_titles(titles)
    titles_so_far.update(titles)
    for title in titles:
        page = get_wiki_page(title)
        if page is None:
            continue
        all_pages.append(page)

        new_pages = recursively_find_all_pages(page.links, titles_so_far)
        for pg in new_pages:
            if pg.title not in [p.title for p in all_pages]:
                all_pages.append(pg)
        titles_so_far.update(page.links)

    return all_pages


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    tokenizer = tiktoken.get_encoding("gpt2")
    return len(tokenizer.encode(text))


def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text


discard_categories = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
                      "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
                      "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
                      "References and notes",]


def extract_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
) -> str:
    """
    Extract the sections of a Wikipedia page, discarding the references and other low information sections
    """
    print('Extracting sections from: ' + title)
    if len(wiki_text) == 0:
        return []

    # find all headings and the coresponding contents
    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont)+4)]
    print(outputs)

    # discard the discard categories, accounting for a tree structure
    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if (
                num_equals <= keep_group_level
            ):
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

        # count the tokens of each section
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    # Create a tuple of (title, section_name, content, number of tokens)
    outputs += [(title, h, c, t) if t < max_len
                else (title, h, reduce_long(c, max_len), count_tokens(reduce_long(c, max_len)))
                for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)]

    return outputs

