#version 150



const vec2 COPRIMES = vec2(2, 3);

vec2 halton(int index) {
vec2 f = vec2(1);
vec2 result = vec2(0);
vec2 ind = vec2(index);

while(ind.x > 0.0 && ind.y > 0.0) {
f /= COPRIMES;
result += f * mod(ind, COPRIMES);
ind = floor(ind / COPRIMES);
}
return result;
}

vec2 calculateJitter() {
return (halton(int(mod((GameTime * 3.0) * 24000.0, 128))) - 0.5) / 1024.0;
}