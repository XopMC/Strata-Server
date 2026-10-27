# Official STREAM reference

Unmodified John D. McCalpin STREAM C5.10 source from the official University of Virginia host:

* Source: https://www.cs.virginia.edu/stream/FTP/Code/stream.c
* Run rules/license reference: https://www.cs.virginia.edu/stream/ref.html
* SHA256: a52bae5e175bea3f7832112af9c085adab47117f7d2ce219165379849231692b

The included license permits use and redistribution, including commercial derivatives, subject to naming/publication rules. Standard results require conformity with the STREAM run rules. Modified or nonconforming variants must be labelled. The source is unchanged; array size/type/NTIMES are normal compiler definitions.

Planned configuration:268435456 doubles per array, three arrays,2GiB each/6GiB total, NTIMES10. Each array exceeds4×the selected-core aggregate LLC; it is also at least4×the observed entire-machine512MiB LLC. STREAM reports best post-warmup algorithmic MB/s: Copy/Scale16bytes per element and Add/Triad24bytes. Those are logical traffic counts, not direct memory-controller/RFO electrical traffic counters and not PCIe/H2D throughput.

Root alone may execute the bounded controller after acquiring model/build locks. The controller never starts CUDA, touches model weights/keeper, changes system cache/THP settings, applies strict membind, installs packages, or stops uploaders. It uses per-child local allocation and actual OpenMP first touch, with process NUMA-map observations.
