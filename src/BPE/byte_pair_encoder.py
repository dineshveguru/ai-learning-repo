"""
A byte pair encoder, a part of the transformer that encodes the given string
input to a format that can be understood by the LLM.
We will implement byte-level BPE that can transform the given string to encoded format
by using a trained encoder.
This code contains both the training part and the inference part.

Process
------------------
* Training
bag of strings (training data [corpus]) -> base vocabulary (individual characters in that bag of strings) -> Merging Process (training) -> New words added to vocabulary -> Assign indexes to new symbols (words)

* Inferencing
Given string tokenize based on the encoder training process -> get index of that particular token -> Return the array
"""

from bpe_helpers import BPEHelpers, get_stats, merge
class BasicTokenizer(BPEHelpers):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        tokens = list(map(int, text_bytes))
        n_before = len(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(tokens)
            max_pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = merge(tokens, idx, max_pair)
            merges[max_pair] = idx
            vocab[idx] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {max_pair} -> {idx} ({vocab[idx]}) had {stats[max_pair]} occurrences")
        self.merges = merges
        self.vocab = vocab
        n_after = len(tokens)
        print(f"Compression Ratio: {n_after / n_before:.2f}X")

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def encode(self, text):
        text_bytes = text.encode("utf-8")
        tokens = list(text_bytes)
        new_tokens = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    new_tokens.append(self.merges[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

t = BasicTokenizer()
train_corups = """
Allan Leslie Walters, CB, CBE, AFC (2 November 1905 19 October 1968) was a senior commander in the Royal Australian Air Force (RAAF). Born in Victoria and raised in Western Australia, he graduated from the Royal Military College, Duntroon, before transferring to the RAAF in 1928. He was one of the service's leading flying instructors and aerobatic pilots between the wars, and was appointed to his first squadron command in 1937. Over the course of World War II, Walters led No. 1 (General Reconnaissance) Squadron in Singapore, No. 1 (Fighter) Wing in Darwin, Northern Territory, No. 72 Wing in Dutch New Guinea, and Northern Command in Papua New Guinea. He was decorated with the Air Force Cross in 1941 for his work with No. 1 Squadron, and mentioned in despatches in 1944 for his service with No. 72 Wing.

Walters was appointed a Commander of the Order of the British Empire in 1946 for his service with Northern Command. Already marked out for senior roles in the post-war RAAF, his positions during the 1950s included Air Officer Commanding (AOC) Southern Area Command, AOC RAAF Overseas Headquarters in London, Head of the Australian Joint Services Staff in Washington, D.C., AOC Home Command, Air Member for Personnel, and AOC Support Command. He was promoted acting air vice marshal in 1952 (substantive in 1954), and appointed a Companion of the Order of the Bath in 1956. Popularly known as "Wally",[1] he was twice a candidate for Chief of the Air Staff, and twice passed over. He retired from the RAAF in 1962 and made his home in Melbourne, where he died in 1968 at the age of sixty-two.

Early life and career
Allan Leslie Walters was born on 2 November 1905 in Ascot Vale, Victoria, to schoolteacher Ferdinand Walters and his wife Edith. The family soon moved to Perth, Western Australia, and Allan completed his education at Perth Modern School, where he joined the cadets. After leaving school and spending eight months in the militia, he entered the Royal Military College, Duntroon, in February 1924.[1] At Duntroon he specialised in field artillery, and excelled at athletics.[2] Graduating as a lieutenant in December 1927, he transferred to the Royal Australian Air Force (RAAF) on 1 February 1928.[1][3] Walters' preferred career path in the military was engineering, and it was only when he failed to gain selection for this field after his graduation that he applied to transfer to the Air Force, which, having no cadet college of its own, had arranged with Duntroon to take one of its artillery specialists each year for secondment as a pilot.[4] He commenced his course at RAAF Point Cook, Victoria, in mid-1928, and graduated as a flying officer in March 1929.[1][4] Walters showed an aptitude for instruction, and after further training was graded an 'A1' flight instructor, a rare distinction. Posted to No. 3 Squadron at RAAF Station Richmond, New South Wales, operating Westland Wapitis, he also made a name for himself performing aerobatics at air shows throughout the country.[1][5] Walters put this particular talent to use in pursuit of his wife-to-be, Jean Manning, stunt flying above All Saints Church, North Parramatta, where her father was rector. Her father officiated at their wedding service in the church on 30 June 1930; their daughter Robin was born in Richmond.[1][6]
Walters was granted a permanent commission in the Air Force in 1930.[2] On 5 January 1931, by now promoted flight lieutenant, he won a trophy in an air obstacle race at the Cootamundra Air Pageant. In May the following year, he took out the NSW Air Derby and Evening News Cup.[7] He temporarily commanded No. 3 Squadron during October 1933, in the absence of Squadron Leader Bill Bostock. At the time, the commanding officer of No. 3 Squadron also held command of RAAF Station Richmond.[8] Walters was posted to Britain in 1936 to attend the Royal Air Force Staff College, Andover, and was promoted to squadron leader in March 1937, while still overseas.[1][9] He also undertook a naval reconnaissance course at RAF Manston. Returning to Australia in May, he took command of No. 22 Squadron in June, flying Hawker Demons and Avro Ansons out of Richmond until February 1938.[2][10]

Between 6 and 23 February 1938, Walters piloted the first overseas flight in an aeroplane designed and built in Australia when he flew the Chief of the Air Staff, Air Vice Marshal Richard Williams, to Singapore in a Tugan Gannet.[11] He returned to Richmond in May 1938 to lead No. 3 Squadron, operating Demons, and again took part in aerobatic displays.[8][12] On 25 October 1938, his Demon crashed in scrub at Tumbi Umbi, New South Wales, when the engine failed shortly after taking off for Richmond, but he was not injured.[13] Completing his Richmond appointment in May 1939, Walters transferred to Melbourne as Director of Staff Duties at RAAF Headquarters.[8][9] Later that month, he joined Group Captain Henry Wrigley as an expert assessor on the panel of an inquiry into a recent series of three Anson accidents; the full report handed down in October found human error the likely explanation for at least one crash and that training on the type followed the syllabus laid down, but that pilots needed more practical experience in dealing with potential in-flight incidents.[14]
Walters' first operational appointment following the outbreak of World War II was as commanding officer of No. 1 (General Reconnaissance) Squadron, which he led to Sembawang, Singapore, in July 1940.[1][15] His promotion to temporary wing commander was announced the same month.[16] He had earlier travelled incognito to Singapore on a Qantas Empire flying boat, which had been specifically requested to deviate from its normal flight path so that he could reconnoiter airfields in the Dutch East Indies.[17] Deployed in response to fears of Japanese expansion in Malaya, No. 1 Squadron was the first Australian unit equipped with Lockheed Hudson light bombers, which were employed primarily for maritime patrol work.[15][18] Walters was awarded the Air Force Cross for his "very active part in all operations" and for training his unit to "a particularly high standard"; the honour was gazetted in the 1941 King's Birthday Honours.[1][19] He succeeded Frank Lukis as commanding officer of RAAF Station Laverton, Victoria, in May the same year, and was promoted acting group captain.[2][20] In May 1942, he joined Allied Air Forces Headquarters, South West Pacific Area (SWPA), in Melbourne as Assistant Director of Operations.[21] He was made a temporary group captain in September, and transferred to Headquarters RAAF Command as senior air staff officer.[22][23]

On 7 October 1942, Walters took command of a new formation, No. 1 (Fighter) Wing, at RAAF Station Richmond. Established to boost the air defence capability of Australia's North-Western Area, the wing comprised three Supermarine Spitfire squadrons that had been transferred from Europe: No. 54 Squadron RAF, No. 452 Squadron RAAF and No. 457 Squadron RAAF. With Wing Commander Clive Caldwell, Australia's top-scoring flying ace of the war, as his wing leader, Walters began deploying aircraft and men to Darwin, Northern Territory, in December, providing a fillip for morale in the region.[24][25] Proudly declaring himself Australia's oldest fighter pilot, Walters was reported as taking every opportunity to join his men in the air.[1][26] He flew as Caldwell's wingman in No. 1 Wing's first major action against the Japanese over Darwin on 2 May 1943.[27] Eight Spitfires crashed and several others made forced landings, for the destruction of one Japanese bomber and five fighters.[28] Walters narrowly avoiding being shot down when he warned Caldwell of an attacking enemy fighter, to the detriment of his own safety. After they landed, Caldwell chided his commander, "You silly old so-and-so. You want to look after your own skin instead of worrying about someone else's!"[27] On 20 June, Walters participated in the wing's most successful combat against the Japanese to that time, personally accounting for one of fourteen raiders claimed by the Spitfires, for the loss of two of their own number.[29] He posted out of Darwin a few days later, having earned the admiration of Caldwell and the rest of the wing's personnel.[30]

Three men in light-coloured uniforms with peaked caps seated on a jeep bonnet
Walters (right) as Officer Commanding No. 72 Wing in New Guinea, December 1943
Walters assumed command of No. 5 Service Flying Training School in Uranquinty, New South Wales, on 30 June 1943, but the next month was posted to Merauke in Dutch New Guinea to take over No. 72 Wing following reassignment of its original commander, Group Captain Charles Eaton.[15][31][32] Comprising No. 84 Squadron (flying CAC Boomerang fighters), No. 86 Squadron (Curtiss P-40 Kittyhawk fighters), and No. 12 Squadron (Vultee A-31 Vengeance dive bombers), No. 72 Wing came under the control of RAAF North-Eastern Area Command, and undertook air defence and patrol tasks in and around western New Guinea.[33] Group Captain Bill Hely assumed command of No. 72 Wing in May 1944, and Walters was appointed Director of Staff Policy and Plans at RAAF Headquarters.[2][34] He was mentioned in despatches on 28 October 1944 for his "Gallant & distinguished service" in North-Eastern Area, the award being promulgated on 9 March 1945.[35][36]

In February 1945, Walters was promoted to acting air commodore and took over from Air Commodore Lukis as Air Officer Commanding (AOC) Northern Command, directing its operations in New Guinea, New Britain and Bougainville until the end of the war.[1][37] Headquartered at Madang in Papua New Guinea, Northern Command had previously been a large mobile formation known as No. 9 (Operational) Group but had evolved into a garrison force, its mobile function supplanted by No. 10 (Operational) Group (later First Tactical Air Force).[38][39] Northern Command's operational formations included No. 71 Wing in northern New Guinea, No. 74 Wing at Port Moresby, and No. 84 Wing on Bougainville.[40] No. 71 Wing, commanded by Group Captain Val Hancock, supported the Australian 6th Division during the Aitape–Wewak campaign, despite ordnance deficiencies that at one stage led to its squadrons arming their Bristol Beauforts with captured Japanese bombs.[41] No. 84 Wing suffered shortages in pilots and equipment during the Bougainville campaign, and morale problems following the end of the war owing to inactivity and the uncertainties of demobilisation; as a result, the wing's commanding officer sent Northern Command headquarters a frank report, the tone of which earned a rebuke from Walters.[42][43] In September, Walters represented the RAAF at the Japanese surrender ceremonies in Wewak.[1]
"""

t.train(train_corups, 400, True)
text = "quadron RAAF"
print(f"Tokens before: {len(list(text.encode('utf-8')))}")
tok = t.encode(text)
print(f"Tokens: {len(tok)}")
text = t.decode(tok)
print(f"decoded text: {text}")