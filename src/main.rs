/*
have many per-base bedgraphs with depth from mosdepth. 
have mappability bed

for each base in each region of mappability bed want to know
1) if mean depth across all samples meets some thresholds. this is current snparcher method
2) how many samples had depth at this base that met thresholds. this is pixy. ie number of comparisons for pi

basic idea here is to get regions from mappability bed and create arrays to store above info

for each region, we fetch the overlapping records from the bedgraph
these records may go outside of our mappability region, so we trim it to fit
then we set the corresponding index in our region array to the depth at that record

example:
mappability bed:
chr1    10  20

bedgraph
chr1    0   12  4 <- all bases here had depth of 4
chr1    12  25  5

create empty depth array of length 9
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[10,11,12,13,14,15,16,17,18,19] <-the bases from mappability bed

when we fetch(chr1:10-20), we get both records from bed graph
for each record, we normalize the start and end so we can index into our array, ie
for [chr1,0,12,4], 
we only care about bases 10,11,12 
which correspond to indices 0,1,2 in our array
so now our array becomes:
[4, 4, 4, 0, 0, 0, 0, 0, 0, 0]
[10,11,12,13,14,15,16,17,18,19] <-the bases from mappability bed

then for [chr1,12,25,5], 
we only care about bases 13,14,15,16,17,18,19
or indices 3,4,5,6,7,8,9
so now our array becomes:
[4, 4, 4, 5, 5, 5, 5, 5, 5, 5]
[10,11,12,13,14,15,16,17,18,19] <-the bases from mappability bed

for subsequent bedgraphs, we can just keep summing arrays element wise,
hopefully limiting memory use

then for snparcher output we can take the mean at each base and decide to keep the base or not

for pixy output, we do a little different. we only keep track if each base passed or not
lets say our threshold for callable site is depth>4
then, for [chr1,0,12,4], none of the bases pass
so our array stays the same:
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[10,11,12,13,14,15,16,17,18,19] <-the bases from mappability bed

then for [chr1,12,25,5], all the bases pass. so we +1 to each base:
 [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
[10,11,12,13,14,15,16,17,18,19] <-the bases from mappability bed

again for subsequent bedgraphs we do this, then sum elementwise
this gives us the number of comparisons for each base


should consider using bams? bedgraphs kinda suck
*/

use csv::ReaderBuilder;
use ndarray::Array1;
use rayon::prelude::*;
use rust_htslib::tbx::{self, Read as TbxRead};
use std::path::Path;
use std::str;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use anyhow::{Result, anyhow};
use std::io::{BufWriter, Write};
use std::fs::File;
#[derive(Debug)]

struct Region {
    chr: String,
    start: usize,
    stop: usize,
}

fn write_bedgraph(file_path: &str, regions: &[Region], total_depths: &[Array1<f64>]) -> Result<()> {
    let file = File::create(file_path)?;
    let mut writer = BufWriter::new(file);

    for (region, depths) in regions.iter().zip(total_depths.iter()) {
        for (i, &depth) in depths.iter().enumerate() {
            let start = region.start + i;
            let end = start + 1;
            writeln!(writer, "{}\t{}\t{}\t{}", region.chr, start, end, depth)?;
        }
    }

    Ok(())
}

fn read_bed(file_path: &Path) -> Result<Vec<Region>> {
    println!("Reading BED file: {:?}", file_path);

    if !file_path.exists() {
        return Err(anyhow!("File not found: {:?}", file_path));
    }

    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_path(file_path)?;

    let mut regions = Vec::new();

    for result in reader.records() {
        match result {
            Ok(record) => {
                // println!("Record: {:?}", record);
                let chr = record[0].to_string();
                let start: usize = record[1].parse()?;
                let stop: usize = record[2].parse()?;
                regions.push(Region { chr, start, stop });
            }
            Err(e) => eprintln!("Error reading record: {:?}", e),
        }
    }

    println!("Finished reading BED file: {} regions found", regions.len());
    Ok(regions)
}

fn process_bedgraph(
    file_path: &Path,
    regions: &[Region],
    accumulated_depths: Arc<Mutex<Vec<Array1<f64>>>>,
) -> Result<()> {
    println!("Reading file: {:?}", file_path);
    let mut reader = tbx::Reader::from_path(file_path)?;

    let mut local_depths: Vec<Array1<f64>> = regions
        .iter()
        .map(|region| Array1::<f64>::zeros(region.stop - region.start))
        .collect();

    for (region_index, region) in regions.iter().enumerate() {
        // prolly dont need to clone here should just use local_depths[i] directly?
        let mut depth = local_depths[region_index].clone();

        let tid = reader.tid(&region.chr)?;
        // println!("Fetching region: {}:{}-{}", region.chr, region.start, region.stop);
        reader.fetch(tid, region.start as u64, region.stop as u64)?;
        // look into using .read() into vec8 instead
        // prolly use noodle? tho this works
        for result in reader.records() {
            match result {
                Ok(record) => {
                    let row_str = str::from_utf8(&record)?;
                    let fields: Vec<&str> = row_str.split('\t').collect();
                    if fields.len() < 4 {
                        eprintln!("Skipping invalid record: {:?}", row_str);
                        continue;
                    }

                    let chrom_start: usize = fields[1].parse()?;
                    let chrom_end: usize = fields[2].parse()?;
                    // score should be int
                    let score: f64 = fields[3].parse()?;

                    let norm_start: usize = if chrom_start < region.start {
                        0
                    } else {
                        chrom_start - region.start
                    };
                    let norm_end: usize = if chrom_end > region.stop {
                        region.stop - region.start
                    } else {
                        chrom_end - region.start
                    };

                    for i in norm_start..norm_end {
                        if i < depth.len() {
                            depth[i] += score; // should just assign not add
                        }
                    }
                }
                Err(e) => eprintln!("Error reading record: {:?}", e),
            }
        }
        local_depths[region_index] = depth;
    }

    let mut accumulated = accumulated_depths.lock().unwrap();
    // iterate accumulated and new arrays together, add em up.
    // have to do it this way b/c cant have 2d array bc each region is diff size
    for (acc, new) in accumulated.iter_mut().zip(local_depths.iter()) {
        *acc += new;
    }

    println!("Finished reading file: {:?}", file_path);
    Ok(())
}

fn main() -> Result<()> {
    let start_time = Instant::now();

    let bed_path = Path::new("data/test.bed");
    let bedgraph_paths = vec![
        Path::new("data/example.bedgraph.gz"),
        Path::new("data/example2.bedgraph.gz"),
        
    ];

    println!("Starting processing");
    let regions = read_bed(bed_path)?;

    // Initialize total depths for each region with zero arrays of len(region)
    // should be int
    let total_depths: Vec<Array1<f64>> = regions
        .iter()
        .map(|region| Array1::<f64>::zeros(region.stop - region.start))
        .collect();

    // Wrap total_depths in arc and mutex for thread safety
    // arc makes sure total_depths or and ref to it doesnt go out of scope till all threads done
    // mutex lets threads lock total_depths when using it
    let total_depths = Arc::new(Mutex::new(total_depths));

    // rayon is magic
    bedgraph_paths.par_iter().try_for_each(|&bedgraph_path| {
        process_bedgraph(bedgraph_path, &regions, Arc::clone(&total_depths))
    })?;

    let duration = start_time.elapsed();
    println!("Elapsed time: {:?}", duration);
    let total_depths = total_depths.lock().unwrap();
    println!("Number of regions: {:?}", total_depths.len());

    // Write BEDGRAPH file
    write_bedgraph("output.bedgraph", &regions, &total_depths)?;
    Ok(())
}
