// Independent output rows are interleaved; each row still sums j in source order.
fn exact_matvec(matrix:&Array2<f32>,input:&[f64],bias:Option<&[f32]>)->Vec<f64>{
    let (rows,cols)=matrix.dim();assert_eq!(cols,input.len());
    if let Some(b)=bias{assert_eq!(b.len(),rows);}
    let values=matrix.as_slice().expect("standard row-major projection");
    let mut output=vec![0.0;rows];let mut row=0;
    while row+4<=rows{
        let r0=&values[row*cols..(row+1)*cols];
        let r1=&values[(row+1)*cols..(row+2)*cols];
        let r2=&values[(row+2)*cols..(row+3)*cols];
        let r3=&values[(row+3)*cols..(row+4)*cols];
        let mut s0=bias.map_or(0.0,|b|b[row] as f64);
        let mut s1=bias.map_or(0.0,|b|b[row+1] as f64);
        let mut s2=bias.map_or(0.0,|b|b[row+2] as f64);
        let mut s3=bias.map_or(0.0,|b|b[row+3] as f64);
        for j in 0..cols{
            let x=input[j];
            s0+=r0[j] as f64*x;s1+=r1[j] as f64*x;
            s2+=r2[j] as f64*x;s3+=r3[j] as f64*x;
        }
        output[row]=s0;output[row+1]=s1;output[row+2]=s2;output[row+3]=s3;row+=4;
    }
    for i in row..rows{
        let values=&values[i*cols..(i+1)*cols];let mut sum=bias.map_or(0.0,|b|b[i] as f64);
        for j in 0..cols{sum+=values[j] as f64*input[j];}
        output[i]=sum;
    }
    output
}
