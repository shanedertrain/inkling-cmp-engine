//! Build tooling: `gen-kernels` (package R2) and `port-map` (generates PORT-MAP checks, not implemented yet).

mod kernels;

use std::path::Path;

use anyhow::bail;

const USAGE: &str = "usage: xtask gen-kernels PACK_DIR OUT_DIR | port-map";

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["gen-kernels", pack, out] => kernels::gen_kernels(Path::new(pack), Path::new(out)),
        ["port-map"] => bail!("not implemented (first porting package)"),
        _ => bail!("{USAGE} (got {args:?})"),
    }
}
