import lmdb

db = lmdb.open(
    "100_val_ood_cat.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

#system_paths = ["CuCO_adslab.traj"]
data_objects = dataset_val_ood_cat
idx = 0

for data_object in data_objects:
    if idx > 100:
        break
#         # Extract Data object
#         #data_objects = read_trajectory_extract_features(a2g, system)
#         #initial_struc = data_object[0]
#         #relaxed_struc = data_object[1]

#         initial_struc.y_init = initial_struc.y # subtract off reference energy, if applicable
#         del initial_struc.y
#         initial_struc.y_relaxed = relaxed_struc.y # subtract off reference energy, if applicable
#         initial_struc.pos_relaxed = relaxed_struc.pos

#         # Filter data if necessary
#         # OCP filters adsorption energies > |10| eV

#         initial_struc.sid = idx  # arbitrary unique identifier 

#         # no neighbor edge case check
#         if initial_struc.edge_index.shape[1] == 0:
#             print("no neighbors", traj_path)
#             continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data_object, protocol=-1))
        txn.commit()
        db.sync()
        print(idx)
        idx += 1

db.close()