FROM rootproject/root-conda

RUN pip3 install --upgrade numpy && \
    pip3 install --upgrade matplotlib && \
    pip3 install uproot coffea jupyter tqdm pandas backports.lzma pyyaml klepto && \
    pip3 install --upgrade tqdm

COPY . /tW_scattering
WORKDIR /tW_scattering
