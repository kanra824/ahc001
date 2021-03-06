use proconio::input;
use std::time::Instant;
use std::cmp;
use std::fmt;
use Direction::*;

const TIME_LIMIT: u128 = 4900;
const LOOP_PER_TIME_CHECK: usize = 100;
const START_TMP: f64 = 50.0;
const END_TMP: f64 = 10.0;
const SZ: i64 = 10000;

#[derive(Clone)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Xorshift {
    seed: u64,
}
impl Xorshift {
    #[allow(dead_code)]
    pub fn new() -> Xorshift {
        Xorshift {
            seed: 0xf0fb588ca2196dac,
        }
    }
    #[allow(dead_code)]
    pub fn with_seed(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }
    #[inline]
    #[allow(dead_code)]
    pub fn next(&mut self) -> u64 {
        self.seed = self.seed ^ (self.seed << 13);
        self.seed = self.seed ^ (self.seed >> 7);
        self.seed = self.seed ^ (self.seed << 17);
        self.seed
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand(&mut self, m: u64) -> u64 {
        self.next() % m
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand_int(&mut self, min: u64, max: u64) -> u64 {
        self.next() % (max - min + 1) + min
    }
    #[inline]
    #[allow(dead_code)]
    // 0.0 ~ 1.0
    pub fn randf(&mut self) -> f64 {
        use std::mem;
        const UPPER_MASK: u64 = 0x3FF0000000000000;
        const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
        let tmp = UPPER_MASK | (self.next() & LOWER_MASK);
        let result: f64 = unsafe { mem::transmute(tmp) };
        result - 1.0
    }
}

#[derive(Clone)]
struct Advertizement {
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
}
impl Advertizement {
    fn area(&self) -> i64 {
        (self.x1 - self.x2).abs() * (self.y1 - self.y2).abs()
    }

    fn cross(&self, adv: &Advertizement) -> bool {
        let mut res = true;
        res = res && ((self.x1 <= adv.x1 && adv.x1 < self.x2) || (adv.x1 < self.x1 && self.x1 <= adv.x2));
        res = res && ((self.y1 <= adv.y1 && adv.y1 < self.y2) || (adv.y1 < self.y1 && self.y1 <= adv.y2));
        res
    }
}
impl fmt::Display for Advertizement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {} {}", self.x1, self.y1, self.x2, self.y2)
    }
}

struct State<'a> {
    n: usize,
    rand: Xorshift,
    score: f64,
    x: &'a Vec<i64>,
    y: &'a Vec<i64>,
    r: &'a Vec<i64>,
    adv: Vec<Advertizement>,
    dir: Vec<Direction>,
    cntupd: i64,
}
impl<'a> State<'a> {
    fn new(n: usize, rand: Xorshift, x: &'a Vec<i64>, y: &'a Vec<i64>, r: &'a Vec<i64>) -> Self {
        let mut state = State{
            n: n,
            rand: rand,
            score: 0.0,
            x: x,
            y: y,
            r: r,
            adv: vec![Advertizement{x1:-1, y1:-1, x2:-1, y2:-1} ;n],
            dir: vec![Left, Right, Up, Down],
            cntupd: 0,
        };

        for i in 0..n {
            state.adv[i] = Advertizement{
                x1: x[i],
                y1: y[i],
                x2: x[i]+1,
                y2: y[i]+1,
            };
        }

        state.score = state.score_all();

        state
    }

    fn update(&mut self, incr: bool, temperature: f64, val: i64) {
        // 変化させるidx
        let i = self.rand.rand_int(0, (self.n-1) as u64) as usize;
        // 変化させる方向
        // 1: ひろげる, 2: ちぢめる
        let sign: i64 = 
            if incr {
                1
            } else {
                self.rand.rand_int(0, 1) as i64 * 2 - 1
            };

        let dir_idx = self.rand.rand_int(0, 3) as usize;
        // どの辺を操作するか
        let dir = self.dir[dir_idx].clone();

        let mut ok = true;
        let prev_area = self.adv[i].area();
        let mut new_score: f64 = self.score - self.score(i, prev_area);

        // 重なってたらやめる
        match dir {
            Left => {
                if sign == 1 {
                    self.adv[i].x1 -= val;
                    if self.adv[i].x1 < 0 {
                        ok = false;
                    }
                    if ok {
                        for j in 0..self.n {
                            // 増えた部分の交差判定
                            if i == j {
                                continue;
                            }
                            if self.adv[i].cross(&self.adv[j]) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        new_score += self.score(i, self.adv[i].area());
                    } else {
                        self.adv[i].x1 += val;
                    } 
                } else {
                    self.adv[i].x1 += val;
                    if self.adv[i].x1 > self.x[i] {
                        ok = false;
                        self.adv[i].x1 -= val;
                    } else {
                        new_score += self.score(i, self.adv[i].area());
                    }
                }
            },
            Right => {
                if sign == 1 {
                    self.adv[i].x2 += val;
                    if self.adv[i].x2 > SZ {
                        ok = false;
                    }
                    if ok {
                        for j in 0..self.n {
                            if i == j {
                                continue;
                            }
                            if self.adv[i].cross(&self.adv[j]) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        new_score += self.score(i, self.adv[i].area());
                    } else {
                        self.adv[i].x2 -= val;
                    }
                } else {
                    self.adv[i].x2 -= val;
                    if self.adv[i].x2 <= self.x[i] {
                        ok = false;
                        self.adv[i].x2 += val;
                    } else {
                        new_score += self.score(i, self.adv[i].area());
                    }
                }
            },
            Up => {
                if sign == 1 {
                    self.adv[i].y1 -= val;
                    if self.adv[i].y1 < 0 {
                        ok = false;
                    }
                    if ok {
                        for j in 0..self.n {
                            if i == j {
                                continue;
                            }
                            if self.adv[i].cross(&self.adv[j]) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        new_score += self.score(i, self.adv[i].area());
                    } else {
                        self.adv[i].y1 += val;
                    }
                } else {
                    self.adv[i].y1 += val;
                    if self.adv[i].y1 > self.y[i] {
                        ok = false;
                        self.adv[i].y1 -= val;
                    } else {
                        new_score += self.score(i, self.adv[i].area());
                    }
                }
            },
            Down => {
                if sign == 1 {
                    self.adv[i].y2 += val;
                    if self.adv[i].y2 > SZ {
                        ok = false;
                    }
                    if ok {
                        for j in 0..self.n {
                            if i == j {
                                continue;
                            }
                            if self.adv[i].cross(&self.adv[j]) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        new_score += self.score(i, self.adv[i].area());
                    } else {
                        self.adv[i].y2 -= val;
                    }
                } else {
                    self.adv[i].y2 -= val;
                    if self.adv[i].y2 <= self.y[i] {
                        ok = false;
                        self.adv[i].y2 += val;
                    } else {
                        new_score += self.score(i, self.adv[i].area());
                    }
                }
            },
        }

        if ok {
            if new_score >= self.score {
                self.cntupd += 1;
                self.score = new_score;
            } else {
                self.revert(i, dir, sign, val);
            }
        }

        //let prob = std::f64::consts::E.powf((new_score - self.score) as f64 / temperature);
        // !!!!!!!!!!!!!!!!!!!! MAXIMIZE !!!!!!!!!!!!!!!!!!!!

        //if self.rand.randf() < prob {
            // update
        //}
    }

    fn revert(&mut self, i: usize, dir: Direction, sign: i64, val: i64) {
        match dir {
            Left => {
                if sign == 1 {
                    self.adv[i].x1 += val;
                } else {
                    self.adv[i].x1 -= val;
                }
            },
            Right => {
                if sign == 1 {
                    self.adv[i].x2 -= val;
                } else {
                    self.adv[i].x2 += val;
                }
            },
            Up => {
                if sign == 1 {
                    self.adv[i].y1 += val;
                } else {
                    self.adv[i].y1 -= val;
                }
            },
            Down => {
                if sign == 1 {
                    self.adv[i].y2 -= val;
                } else {
                    self.adv[i].y2 += val;
                }
            },
        }
    }

    // 10^9をかける手前までのスコアを計算
    fn score_all(&mut self) -> f64 {
        let mut score: f64 = 0.0;
        for i in 0..self.n {
            if self.adv[i].x1 == -1 {
                continue;
            }
            let nows = self.adv[i].area();
            score += self.score(i, nows) as f64;
        }
        score
    }

    // 広告ひとつのスコアを計算
    fn score(&self, i: usize, s: i64) -> f64 {
        // 1 - (1 - min(ri, si) / max(ri, si)) ^ 2
        1.0 - (1.0 - cmp::min(self.r[i], s) as f64 / cmp::max(self.r[i], s) as f64).powi(2)
    }
}

fn init(state: &mut State, start: &Instant, time: u128) {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < time {
        let temperature: f64 = START_TMP + (END_TMP - START_TMP) * (elapsed_time as f64) / (TIME_LIMIT as f64);
        for _ in 0..LOOP_PER_TIME_CHECK {
            state.update(true, temperature, 100);
        }
        elapsed_time = start.elapsed().as_millis();
    }
}

fn simulate(state: &mut State, start: &Instant) {
    let mut elapsed_time = start.elapsed().as_millis();
    while elapsed_time < TIME_LIMIT {
        let temperature: f64 = START_TMP + (END_TMP - START_TMP) * (elapsed_time as f64) / (TIME_LIMIT as f64);
        for _ in 0..LOOP_PER_TIME_CHECK {
            state.update(true, temperature, 1);
        }
        elapsed_time = start.elapsed().as_millis();
    }
}

fn main() {
    let start = Instant::now();

    input! {
        inN: usize,
        input: [[i64; 3]; inN],
    }

    let mut inx: Vec<i64> = vec![0; inN];
    let mut iny: Vec<i64> = vec![0; inN];
    let mut inr: Vec<i64> = vec![0; inN];

    for i in 0..inN {
        inx[i] = input[i][0];
        iny[i] = input[i][1];
        inr[i] = input[i][2];
    }

    let mut sel = vec![true; inN];

    for i in 0..inN {
        for j in 0..i {
            if inx[i] == inx[j] && iny[i] == iny[j] {
                if inr[i] <= inr[j] {
                    sel[j] = false;
                } else {
                    sel[i] = false;
                }
            }
        }
    }

    let mut x: Vec<i64> = Vec::new();
    let mut y: Vec<i64> = Vec::new();
    let mut r: Vec<i64> = Vec::new();
    let mut sel_idx: Vec<usize> = Vec::new();
    let mut not_sel_idx: Vec<usize> = Vec::new();

    for i in 0..inN {
        if sel[i] {
            x.push(inx[i]);
            y.push(iny[i]);
            r.push(inr[i]);
            sel_idx.push(i);
        } else {
            not_sel_idx.push(i);
        }
    }

    let n = x.len();
    
    let seed = start.elapsed().as_nanos() as u64;
    let rand = Xorshift::with_seed(seed);
    let mut state = State::new(n, rand, &x, &y, &r);
    init(&mut state, &start, TIME_LIMIT / 300);
    simulate(&mut state, &start);

    let mul = 1000000000;
    eprintln!("cntupd: {}", state.cntupd);
    eprintln!("saved score: {}", state.score / n as f64 * mul as f64);
    eprintln!("calculated score: {}", state.score_all() / n as f64 * mul as f64);



    // print answer
    if n == inN {
        for i in 0..n {
            println!("{}", state.adv[i]);
        }
    } else {
        panic!("入力に同じ(x, y)が存在")
    }

}