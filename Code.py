# Importing necessary libraries
from Bio import Entrez, SeqIO
import numpy as np
from scipy.linalg import svd
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Function to fetch sequences from NCBI GenBank
def fetch_genbank_sequence(genbank_id):
    Entrez.email = "vikash1221149@gmail.com"  # Replace with your email
    try:
        handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        return str(record.seq)
    except Exception as e:
        print(f"Error fetching GenBank ID {genbank_id}: {e}")
        return None

# Z-Curve method to map sequence to X, Y, Z vectors
def z_curve_mapping(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = [mapping.get(nuc, 0) for nuc in sequence]

    x_vector = [seq[i] * np.cos(2 * np.pi * i / len(seq)) for i in range(len(seq))]
    y_vector = [seq[i] * np.sin(2 * np.pi * i / len(seq)) for i in range(len(seq))]
    z_vector = [seq[i] for i in range(len(seq))]

    return np.array(x_vector), np.array(y_vector), np.array(z_vector)

# Linear Predictive Coding (LPC) method for feature extraction
def lpc(sequence, window_size):
    features = []
    for i in range(0, len(sequence) - window_size + 1, window_size):
        window = sequence[i:i+window_size]
        lpc_coeffs = np.polyfit(np.arange(len(window)), window, 1)
        features.extend(lpc_coeffs)
    return np.array(features)

# Singular Value Decomposition (SVD) for dimensionality reduction
def apply_svd(features):
    U, S, Vt = svd(features, full_matrices=False)
    return U, S, Vt

# Fetch sequences from GenBank and process
def prepare_data():
    # GenBank IDs for coronavirus and influenza
    genbank_ids = {
        "coronavirus": ['PQ655348.1', 'PQ655346.1', 'PQ655345.1', 'PQ655344.1', 'PQ655343.1', 'PQ651386.1', 'PQ651385.1', 'PQ651384.1', 'PQ651382.1', 'PQ651381.1', 'PQ651380.1', 'PQ651379.1', 'PQ651378.1', 'PQ651377.1', 'PQ651376.1', 'PQ651375.1', 'PQ651374.1', 'PQ651370.1', 'PQ651369.1', 'PQ651368.1', 'PQ651367.1', 'PQ651366.1', 'PQ651365.1', 'PQ651364.1', 'PQ651363.1', 'PQ651361.1', 'PQ651360.1', 'PQ651359.1', 'PQ651358.1', 'PQ651357.1', 'PQ651356.1', 'PQ651355.1', 'PQ651354.1', 'PQ651353.1', 'PQ651352.1', 'PQ651351.1', 'PQ651350.1', 'PQ651349.1', 'PQ651348.1', 'PQ651347.1', 'PQ651344.1', 'PQ651343.1', 'PQ651342.1', 'PQ651341.1', 'PQ651340.1', 'PQ651339.1', 'PQ651338.1', 'PQ651337.1', 'PQ651336.1', 'PQ651334.1', 'PQ651332.1', 'PQ651331.1', 'PQ651330.1', 'PQ651329.1', 'PQ651328.1', 'PQ651325.1', 'PQ651324.1', 'PQ651323.1', 'PQ651322.1', 'PQ651318.1', 'PQ651317.1', 'PQ651316.1', 'PQ651315.1', 'PQ651314.1', 'PQ651313.1', 'PQ651312.1', 'PQ651309.1', 'PQ651308.1', 'PQ651258.1', 'PQ651253.1', 'PQ651230.1', 'PQ651226.1', 'PQ651218.1', 'PQ651171.1', 'PQ651163.1', 'PQ651158.1', 'PQ651097.1', 'PQ651005.1', 'PQ650915.1', 'PQ650820.1', 'PQ650812.1', 'OR378605.1', 'OR378604.1', 'OR378603.1', 'OR378602.1', 'OR378600.1', 'OR378599.1', 'OR378598.1', 'OR378597.1', 'OR378596.1', 'OR378593.1', 'OR378592.1', 'OR378591.1', 'OR378590.1', 'OR378589.1', 'OR378587.1', 'OR378586.1', 'OR186163.1', 'OR186156.1', 'OR186136.1', 'OR186132.1', 'OR186119.1', 'OR186107.1', 'OR186059.1', 'OR185891.1', 'OR185890.1', 'OR185887.1', 'OR185885.1', 'OR185884.1', 'OR185882.1', 'OR185875.1', 'OR185874.1', 'OR185872.1', 'OR185870.1', 'OR185868.1', 'OR185866.1', 'OR185855.1', 'OR185853.1', 'OR185851.1', 'OR185848.1', 'OR185846.1', 'OR185844.1', 'OR185843.1', 'OR185842.1', 'OR185841.1', 'OR185839.1', 'OR185838.1', 'OR185837.1', 'OR185836.1', 'OR185835.1', 'OR185834.1', 'OR185833.1', 'OR185832.1', 'OR185831.1', 'OR185829.1', 'OR185828.1', 'OR185827.1', 'OR185826.1', 'OR185825.1', 'OR185823.1', 'OR185822.1', 'OR185821.1', 'OR185820.1', 'OR185817.1', 'OR185816.1', 'OR185815.1', 'OR185811.1', 'OR185810.1', 'OR185809.1', 'OR185808.1', 'OR185805.1', 'OR185802.1', 'OR185798.1', 'OR185797.1', 'OR185796.1', 'OR185794.1', 'OR185792.1', 'OR185790.1', 'OR185789.1', 'OR185788.1', 'OR185787.1', 'OR185786.1', 'OR185783.1', 'OR185781.1', 'OR185780.1', 'OR185778.1', 'OR185772.1', 'OR185771.1', 'OR185770.1', 'OR185769.1', 'OR185768.1', 'OR185765.1', 'OR185764.1', 'OR185763.1', 'OR185761.1', 'OR185760.1', 'OR185759.1', 'OR185758.1', 'OR185757.1', 'OR185756.1', 'OR185755.1', 'OR185754.1', 'OR185753.1', 'OR185752.1', 'OR185751.1', 'OR185750.1', 'OR185749.1', 'OR185748.1', 'OR185747.1', 'OR185746.1', 'OR185745.1', 'OR185744.1', 'OR185743.1', 'OR185742.1', 'OR185741.1', 'OR185739.1', 'OR185738.1', 'OR185737.1', 'OR185736.1', 'OR185732.1', 'OR185731.1', 'OR185730.1', 'OR185729.1', 'OR185728.1', 'OR185727.1', 'OR185726.1', 'OR185725.1', 'OR185724.1', 'OR185723.1', 'OR185722.1', 'OR185721.1', 'OR185720.1', 'OR185718.1', 'OR185713.1', 'OR185711.1', 'OR185710.1', 'OR185709.1', 'OR185708.1', 'OR185705.1', 'OR185704.1', 'OR185703.1', 'OR185702.1', 'OR185701.1', 'OR185700.1', 'OR185698.1', 'OR185695.1', 'OR185692.1', 'OR185691.1', 'OR185690.1', 'OR185687.1', 'OR185686.1', 'OR185684.1', 'OR185683.1', 'OR185682.1', 'OR185681.1', 'OR185680.1', 'OR185677.1', 'OR185676.1', 'OR185673.1', 'OR185672.1', 'OR185671.1', 'OR185670.1', 'OR185668.1', 'OR185666.1', 'OR185665.1', 'OR185664.1', 'OR185663.1', 'OR185662.1', 'OR185661.1', 'OR185658.1', 'OR185657.1', 'OR185656.1', 'OR185655.1', 'OR185654.1', 'OR185653.1', 'OR185652.1', 'OR185651.1', 'OR185650.1', 'OR185649.1', 'OR185648.1', 'OR185647.1', 'OR185646.1', 'OR185645.1', 'OR185644.1', 'OR185643.1', 'OR185642.1', 'OR185641.1', 'OR185640.1', 'OR185639.1', 'OR185638.1', 'OR185637.1', 'OR185636.1', 'OR185635.1', 'OR185634.1', 'OR185633.1', 'OR185632.1', 'OR185631.1', 'OR185630.1', 'OR185629.1', 'OR185628.1', 'OR185627.1', 'OR185626.1', 'OR185625.1', 'OR185624.1', 'OR185623.1', 'OR185622.1', 'OR185621.1', 'OR185620.1', 'OR185619.1', 'OR185618.1', 'OR185617.1', 'OR185612.1', 'MZ960757.1', 'MZ960573.1', 'MZ960298.1', 'MZ960183.1', 'PQ650450.1', 'PQ650449.1', 'PQ650448.1', 'PQ650446.1', 'PQ650444.1', 'PQ650442.1', 'PQ650441.1', 'PQ650440.1', 'PQ650439.1', 'PQ650415.1', 'PQ650414.1', 'PQ650413.1', 'PQ650412.1', 'PQ650411.1', 'PQ650410.1', 'PQ650409.1', 'PQ650408.1', 'PQ650407.1', 'PQ650405.1', 'PQ650404.1', 'PQ650401.1', 'PQ650400.1', 'PQ650399.1', 'PQ650219.1', 'PQ650216.1', 'PQ650215.1', 'PQ650208.1', 'PQ650207.1', 'PQ650206.1', 'PQ650205.1', 'PQ650204.1', 'PQ650203.1', 'PQ650202.1', 'PQ650201.1', 'PQ650200.1', 'PQ650199.1', 'PQ650198.1', 'PQ650197.1', 'PQ650196.1', 'PQ650195.1', 'PQ650194.1', 'PQ650193.1', 'PQ650192.1', 'PQ650191.1', 'PQ650190.1', 'PQ650189.1', 'PQ650188.1', 'PQ650187.1', 'PQ650186.1', 'PQ650185.1', 'PQ650184.1', 'PQ650183.1', 'PQ650182.1', 'PQ650181.1', 'PQ650180.1', 'PQ650179.1', 'PQ650178.1', 'PQ650177.1', 'PQ650176.1', 'PQ650175.1', 'PQ650174.1', 'PQ650173.1', 'PQ650172.1', 'PQ650171.1', 'PQ650170.1', 'PQ650169.1', 'PQ650168.1', 'PQ650167.1', 'PQ650166.1', 'PQ650165.1', 'PQ650164.1', 'PQ650163.1', 'PQ650162.1', 'PQ650161.1', 'PQ650160.1', 'PQ650159.1', 'PQ650158.1', 'PQ650157.1', 'PQ650156.1', 'PQ650155.1', 'PQ650154.1', 'PQ650153.1', 'PQ650152.1', 'PQ650151.1', 'PQ650150.1', 'PQ650149.1', 'PQ650148.1', 'PQ650147.1', 'PQ650146.1', 'PQ650145.1', 'PQ650144.1', 'PQ650143.1', 'PQ650142.1', 'PQ650141.1', 'PQ650140.1', 'PQ650139.1', 'PQ650138.1', 'PQ650137.1', 'PQ650136.1', 'PQ650135.1', 'PQ650134.1', 'PQ650133.1', 'PQ650132.1', 'PQ650129.1', 'PQ650128.1', 'PQ650127.1', 'PQ649798.1', 'PQ649769.1', 'PQ649602.1', 'PQ649587.1', 'PQ649576.1', 'PQ649453.1', 'PQ649356.1', 'PQ649352.1', 'PQ649345.1', 'PQ649385.1', 'PQ649245.1', 'PQ649235.1', 'PQ649217.1', 'PQ649174.1', 'PQ649029.1', 'PQ648992.1', 'PQ649130.1', 'PQ649122.1', 'PQ649108.1', 'PQ649099.1', 'PQ648790.1', 'PQ648940.1', 'PQ648934.1', 'PQ648927.1', 'PQ648913.1', 'PQ648753.1', 'PQ648751.1', 'PQ648735.1', 'PQ648884.1', 'PQ648705.1', 'PQ648832.1', 'PQ648662.1', 'PQ648644.1', 'PQ648639.1', 'PQ648626.1', 'PQ648622.1', 'PQ648540.1', 'PQ648464.1', 'PQ648439.1', 'PQ648370.1', 'PQ648257.1', 'PQ648318.1', 'PQ648279.1', 'PQ648272.1', 'PQ648165.1', 'PQ648144.1', 'PQ648116.1', 'PQ648082.1', 'PQ648018.1', 'PQ647851.1', 'PQ647803.1', 'PQ647975.1', 'PQ647953.1', 'PQ647768.1', 'PQ647670.1', 'PQ647621.1', 'PQ647594.1', 'PQ647587.1', 'PQ647507.1', 'PQ647489.1', 'PQ647416.1', 'PQ647186.1', 'PQ647218.1', 'PQ645560.1', 'PQ645551.1', 'PQ645432.1', 'PQ645224.1', 'PQ645216.1', 'PQ645213.1', 'PQ645198.1', 'PQ645194.1', 'PQ645191.1', 'PQ645190.1', 'PQ645188.1', 'PQ645185.1', 'PQ643720.1', 'PQ643719.1', 'PQ643718.1', 'PQ643717.1', 'PQ643716.1', 'PQ643715.1', 'PQ643714.1', 'PQ643712.1', 'PQ643711.1', 'PQ643710.1', 'PQ643709.1', 'PQ643708.1', 'PQ643707.1', 'PQ643706.1', 'PQ643705.1', 'PQ643704.1', 'PQ643703.1', 'PQ643702.1', 'PQ643701.1', 'PQ643699.1', 'PQ643698.1', 'PQ643697.1', 'PQ643696.1', 'PQ643695.1', 'PQ643693.1', 'PQ643692.1', 'PQ643691.1', 'PQ643690.1', 'PQ643689.1', 'PQ643688.1', 'PQ643685.1', 'PQ643684.1', 'PQ643683.1', 'PQ643682.1', 'PQ643681.1', 'PQ643680.1', 'PQ643679.1', 'PQ643678.1', 'PQ643677.1', 'PQ643676.1', 'PQ643675.1', 'PQ643674.1', 'PQ643673.1', 'PQ643672.1', 'PQ643671.1', 'PQ643670.1', 'PQ643669.1', 'PQ641424.1', 'PQ641422.1', 'PQ641421.1', 'PQ641420.1', 'PQ641419.1', 'PQ641418.1', 'PQ641417.1', 'PQ641416.1', 'PQ641415.1', 'PQ641413.1', 'PQ641412.1', 'PQ641411.1', 'PQ641410.1', 'PQ641408.1', 'PQ641407.1', 'PQ641405.1', 'PQ641403.1', 'PQ641402.1', 'PQ641401.1', 'PQ641400.1', 'PQ641398.1', 'PQ641397.1', 'PQ641396.1', 'PQ641395.1', 'PQ641394.1', 'PQ641393.1', 'PQ641392.1', 'PQ641390.1', 'PQ641389.1', 'PQ641388.1', 'PQ641386.1', 'PQ641385.1', 'PQ641384.1', 'PQ641383.1', 'PQ641382.1', 'PQ641378.1', 'PQ641376.1', 'PQ641373.1', 'PQ641372.1', 'PQ641371.1', 'PQ641370.1', 'PQ641369.1', 'PQ641367.1', 'PQ641366.1', 'PQ641365.1', 'PQ641364.1', 'PQ641363.1', 'PQ641362.1', 'PQ641360.1', 'PQ641359.1', 'PQ641358.1', 'PQ641356.1', 'PQ641355.1', 'PQ641354.1', 'PQ641352.1', 'PQ641351.1', 'PQ641350.1', 'PQ641349.1', 'PQ641347.1', 'PQ641345.1', 'PQ641221.1', 'PQ641220.1', 'PQ641219.1', 'PQ641218.1', 'PQ641217.1', 'PQ641216.1', 'PQ641215.1', 'PQ641212.1', 'PQ641211.1', 'PQ641209.1', 'PQ641206.1', 'PQ641204.1', 'PQ641203.1', 'PQ641202.1', 'PQ641201.1', 'PQ641200.1', 'PQ641199.1', 'PQ641198.1', 'PQ641197.1', 'PQ641196.1', 'PQ641195.1', 'PQ641194.1', 'PQ641191.1', 'PQ641190.1', 'PQ641187.1', 'PQ641186.1', 'PQ641185.1', 'PQ641184.1', 'PQ641181.1', 'PQ641178.1', 'PQ641177.1', 'PQ641176.1', 'PQ641175.1', 'PQ641173.1', 'PQ641171.1', 'PQ641169.1', 'PQ641167.1', 'PQ641166.1', 'PQ641165.1', 'PQ641164.1', 'PQ641162.1', 'PQ641161.1', 'PQ641158.1', 'PQ641157.1', 'PQ641155.1', 'PQ641151.1', 'PQ641150.1', 'PQ641149.1', 'PQ641148.1'],
        "influenza": ['AL645882.2', 'OR371530.1', 'OR467307.1', 'OR467305.1', 'OR467303.1', 'OR467302.1', 'OR467300.1', 'OR467299.1', 'OR467298.1', 'OR467297.1', 'OR467295.1', 'OR467293.1', 'OR467292.1', 'OR467291.1', 'OR467290.1', 'OR467289.1', 'OR467287.1', 'OR467284.1', 'OR467283.1', 'OR467282.1', 'OR467279.1', 'OR467280.1', 'OR467281.1', 'OR467277.1', 'OR467276.1', 'OR467275.1', 'OR467274.1', 'OR467273.1', 'OR467272.1', 'OR467271.1', 'OR467269.1', 'OR467268.1', 'OR467270.1', 'OR467267.1', 'OR467266.1', 'OR467264.1', 'OR467262.1', 'OR467260.1', 'OR467261.1', 'OR467258.1', 'OR467256.1', 'OR467254.1', 'OR467255.1', 'OR467253.1', 'OR467252.1', 'OR467251.1', 'OR467250.1', 'PP886719.1', 'PP886717.1', 'PP886716.1', 'OR192110.1', 'OR192102.1', 'OR192100.1', 'OR192084.1', 'OR192079.1', 'OR192045.1', 'OR192043.1', 'PP801323.1', 'PP801322.1', 'PP801321.1', 'PP801320.1', 'OR053528.1', 'OR053527.1', 'OR053525.1', 'OR053524.1', 'OR053523.1', 'OR053521.1', 'OR053520.1', 'OR053517.1', 'OR053518.1', 'OR053515.1', 'OR053516.1', 'OR053514.1', 'OR053513.1', 'OR053512.1', 'OR053509.1', 'OR053506.1', 'OR053505.1', 'OR053504.1', 'OR053503.1', 'OR053501.1', 'OR053499.1', 'OR053497.1', 'OR053494.1', 'OR053492.1', 'OR053490.1', 'OR053489.1', 'OR053487.1', 'OR053485.1', 'OR053209.1', 'OR053207.1', 'OR053206.1', 'OR053202.1', 'OR053203.1', 'OR053201.1', 'OR053198.1', 'OR053196.1', 'OR053194.1', 'OR053192.1', 'OR053191.1', 'OR053189.1', 'OR053187.1', 'OR053186.1', 'OR053185.1', 'OR053183.1', 'OR053180.1', 'OR053179.1', 'OR053178.1', 'OR053175.1', 'OR053176.1', 'OR053174.1', 'OR053173.1', 'OR053172.1', 'OR053171.1', 'OR053170.1', 'OR053167.1', 'OR053168.1', 'OR053151.1', 'OR053148.1', 'OR053145.1', 'OR053144.1', 'OR053143.1', 'OR053138.1', 'OR053133.1', 'OR053132.1', 'OR053125.1', 'OR053121.1', 'OR053117.1', 'OR053115.1', 'OR053116.1', 'OR053112.1', 'OR053110.1', 'OR053108.1', 'OR053107.1', 'OR053106.1', 'OR053105.1', 'OR053097.1', 'OR053094.1', 'OR053093.1', 'OR053090.1', 'OR053089.1', 'OR053087.1', 'OR053086.1', 'OR053083.1', 'OR053082.1', 'OR053084.1', 'OR053081.1', 'OR053080.1', 'OR053073.1', 'OR053053.1', 'OR053048.1', 'OR053047.1', 'OR053045.1', 'OR053007.1', 'OR053000.1', 'OR052998.1', 'OR052995.1', 'OR052992.1', 'OR052993.1', 'OR052971.1', 'OR052961.1', 'OR052955.1', 'OR052941.1', 'OR052938.1', 'OR052934.1', 'OR052932.1', 'OR052931.1', 'OR052860.1', 'PP737254.1', 'PP737253.1', 'LC654459.1', 'LC654447.1', 'LC654445.1', 'PP716096.1', 'PP716095.1', 'PP716094.1', 'PP716093.1', 'PP522473.1', 'PP522472.1', 'PP522470.1', 'PP522469.1', 'PP522466.1', 'PP522465.1', 'PP522464.1', 'PP522463.1', 'PP522461.1', 'PP522460.1', 'PP522459.1', 'PP522458.1', 'PP522457.1', 'PP522456.1', 'PP522455.1', 'PP522454.1', 'PP522453.1', 'PP522452.1', 'PP522451.1', 'PP522450.1', 'PP522449.1', 'PP522446.1', 'PP522445.1', 'PP522444.1', 'PP522443.1', 'PP522442.1', 'PP522440.1', 'PP522439.1', 'PP522438.1', 'PP522437.1', 'PP522435.1', 'PP522433.1', 'PP522432.1', 'PP522431.1', 'PP522427.1', 'PP522426.1', 'PP522425.1', 'PP522424.1', 'PP522422.1', 'PP522421.1', 'PP522419.1']
    }

    sequences = []
    labels = []
    window_size = 50  # Define a fixed window size for LPC

    for label, ids in genbank_ids.items():
        for genbank_id in ids:
            print(f"Fetching data for GenBank ID: {genbank_id}")
            sequence = fetch_genbank_sequence(genbank_id)

            if sequence:
                # Z-Curve Mapping
                x_vector, y_vector, z_vector = z_curve_mapping(sequence)

                # LPC Feature Extraction
                lpc_features = lpc(x_vector, window_size=window_size)

                # Combine features (use truncated y and z vectors to match LPC features)
                combined_features = np.vstack((lpc_features, y_vector[:len(lpc_features)], z_vector[:len(lpc_features)])).T

                # Apply SVD for dimensionality reduction
                _, _, Vt = apply_svd(combined_features)

                # Use Vt as the final feature representation
                sequences.append(Vt.flatten())  # Flatten to ensure uniform feature length
                labels.append(1 if label == "coronavirus" else 0)

    return np.array(sequences), np.array(labels)

# Train and evaluate the SVM model
def train_svm(sequences, labels):
    # Scale features
    scaler = StandardScaler()
    sequences = scaler.fit_transform(sequences)

    # Initialize SVM classifier
    model = svm.SVC(kernel='linear')

    # Stratified K-Fold cross-validation
    n_splits = 200
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, sequences, labels, cv=skf)

    # Output results
    print(f" Accuracy: {np.mean(cv_scores)}")
    print(f"Cross-Validation Scores: {cv_scores}")
    plt.plot(cv_scores, marker='o')
    plt.title(f"Stratified Cross-validation Scores :{n_splits} " )
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.show()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Main Execution
if __name__ == "__main__":
    sequences, labels = prepare_data()
    if sequences.size > 0 and labels.size > 0:
        train_svm(sequences, labels)
    else:
        print("No valid data to train the model.")
