import pickle
import boto3
from tqdm.notebook import tqdm


class BPE:
    def __init__(self):
        self.vocab = {}
        self.bpe_codes = {}

    def train(self, corpus, vocab_size, max_merges):
        """
        Trains the model on a given corpus to build the vocabulary and create BPE codes.

        Parameters:
            corpus (list): A list of sentences representing the corpus.

        Returns:
            None
        """
        # Build vocabulary
        for sentence in tqdm(corpus, desc="Building vocabulary"):
            for word in sentence.split():
                if word not in self.vocab:
                    self.vocab[word] = 0
                self.vocab[word] += 1

        # split vocabulary into chars
        new_vocab = {}
        for word, freq in tqdm(self.vocab.items(), desc="Splitting vocabulary"):
            new_word = ' '.join(word)
            new_vocab[new_word] = freq
        self.vocab = new_vocab

        # Create BPE codes
        for _ in tqdm(range(max_merges), desc="Creating BPE codes"):
            if len(self.vocab) >= vocab_size:
                print("Vocab size reached.")
                break
            pairs = self.get_pairs()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] <= 1:
                break
            self.merge_tokens(*best_pair)

    def get_pairs(self):
        """
        Generates pairs of consecutive symbols from the given vocabulary and their frequencies.

        Returns:
            dict: A dictionary containing pairs of symbols as keys and their frequencies as values.
        """
        pairs = {}
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += freq
        return pairs

    def merge_tokens(self, token1, token2):
        """
        Merges two tokens and updates the BPE codes and vocabulary.

        Parameters:
            token1 (str): The first token to be merged.
            token2 (str): The second token to be merged.

        Returns:
            None
        """
        new_token = token1 + token2
        self.bpe_codes[token1 + ' ' + token2] = new_token
        self.update_vocab(token1, token2, new_token)

    def update_vocab(self, token1, token2, new_token):
        """
        Update the vocabulary by replacing occurrences of a pair of tokens with a new token.

        Args:
            token1 (str): The first token in the pair.
            token2 (str): The second token in the pair.
            new_token (str): The new token to replace the pair.

        Returns:
            None
        """
        updated_vocab = {}
        for word, freq in self.vocab.items():
            updated_word = word.replace(token1 + ' ' + token2, new_token)
            updated_vocab[updated_word] = freq
        self.vocab = updated_vocab

    def encode(self, sentence):
        """
        Encode a sentence by splitting it into words and encoding each word individually.

        Parameters:
            sentence (str): The sentence to be encoded.

        Returns:
            list: A list of encoded words from the sentence.
        """
        encoded_sentence = []
        words = sentence.split()
        for word in words:
            encoded_word = self.encode_word(word)
            encoded_sentence.extend(encoded_word)
        return encoded_sentence

    def encode_word(self, word):
        """
        Encodes a given word into a list of subword units using byte pair encoding (BPE).

        Parameters:
            word (str): The word to be encoded.

        Returns:
            list: A list of subword units representing the encoded word.
        """
        if word in self.bpe_codes.values():
            return [word]

        # Split the word into subword units
        subwords = []
        i = 0
        while i < len(word):
            found_subword = False
            # Try to find the longest matching subword from the end of the word
            for j in range(len(word), i, -1):
                subword_candidate = word[i:j]
                if subword_candidate in self.bpe_codes.values():
                    subwords.append(subword_candidate)
                    i = j
                    found_subword = True
                    break
            if not found_subword:
                subwords.append(word[i])
                i += 1
        return subwords

    def decode(self, encoded_sentence):
        """
        Decode the given encoded sentence using the BPE codes.

        Parameters:
        - encoded_sentence (str): The sentence to be decoded.

        Returns:
        - str: The decoded sentence.
        """
        decoded_sentence = []
        i = 0
        while i < len(encoded_sentence):
            token = encoded_sentence[i]
            if token in self.bpe_codes.values():
                decoded_sentence.append(token)
            else:
                j = i + 1
                while j < len(encoded_sentence) and encoded_sentence[j] in self.bpe_codes.values():
                    token += encoded_sentence[j]
                    j += 1
                decoded_sentence.append(token)
                i = j - 1
            i += 1
        return " ".join(decoded_sentence)

    def save(self, file_path):
        """
        Save the vocabulary and BPE codes to a file.

        :param file_path: The path to the file where the data will be saved.
        :type file_path: str
        """
        with open(file_path, "wb") as file:
            pickle.dump((self.vocab, self.bpe_codes), file)

    def save_to_s3(self, bucket_name, object_key):
        """
        Save the contents of the file 'bpe_model.pkl' to an S3 bucket.

        Parameters:
            bucket_name (str): The name of the S3 bucket.
            object_key (str): The key that identifies the object in the bucket.

        Returns:
            None
        """
        s3 = boto3.client('s3')
        with open("bpe_model.pkl", "wb") as file:
            pickle.dump((self.vocab, self.bpe_codes), file)
        s3.upload_file("bpe_model.pkl", bucket_name, object_key)

    def get_vocab(self):
        """
        Get the vocabulary.

        Returns:
            The vocabulary.
        """
        return self.vocab

    def get_bpe_codes(self):
        """
        Get the BPE codes.

        Returns:
            The BPE codes.
        """
        return self.bpe_codes

    def get_stats(self):
        """
        Returns statistics about the current state of the object.

        Parameters:
        - None

        Returns:
            - stats (dict): A dictionary containing the following statistics:
            - num_unique_tokens (int): The number of unique tokens in the vocabulary.
            - num_merges_performed (int): The number of merges performed in the BPE codes.
            - avg_encoded_token_length (float): The average length of the encoded tokens.
        """
        num_unique_tokens = len(self.vocab)
        num_merges_performed = len(self.bpe_codes)
        avg_encoded_token_length = sum(
            len(token) for token in self.bpe_codes.values()) / num_merges_performed

        stats = {
            "num_unique_tokens": num_unique_tokens,
            "num_merges_performed": num_merges_performed,
            "avg_encoded_token_length": avg_encoded_token_length
        }

        return stats


if __name__ == '__main__':
    from tqdm import tqdm
    bpe = BPE()
    with open('titles.txt', 'r') as f:
        corpus = f.read().splitlines()
    bpe.train(corpus[:3000000], max_merges=5000, vocab_size=10000000)
    sent = bpe.encode('software engineer')
    print(sent)
    # word = bpe.encode_word('software')
    # print(word)
    # word2 = bpe.encode_word('softwareengineer')
    # print(word2)
