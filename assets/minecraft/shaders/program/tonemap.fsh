#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform sampler2D blursampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

in vec2 texCoord;




  #define EXPOSURE 1.45 //[1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 4.4 4.6 4.8 5.0 5.2 5.4 5.6 5.8 6.0 6.2 6.4 6.6 6.8 7.0 7.2 7.4 7.6 7.8 8.0 8.2 8.4 8.6 8.8 9.0 9.2 9.4 9.6 9.8 10.0 10.2 10.4 10.6 10.8 11.0 11.2 11.4 11.6 11.8 12.0 12.2 12.4 12.6 12.8 13.0 13.2 13.4 13.6 13.8 14.0 14.2 14.4 14.6 14.8 15.0 15.2 15.4 15.6 15.8 16.0]
  #define TONEMAP_WHITE_CURVE 2.0 //[1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0]
  #define TONEMAP_LOWER_CURVE 1.2 //[0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00]
  #define TONEMAP_UPPER_CURVE 1.3 //[0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00]
  #define SATURATION 0.25 // Negative values desaturates colors, Positive values saturates color, 0 is no change [-1.0 -0.98 -0.96 -0.94 -0.92 -0.9 -0.88 -0.86 -0.84 -0.82 -0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.48 -0.46 -0.44 -0.42 -0.4 -0.38 -0.36 -0.34 -0.32 -0.3 -0.28 -0.26 -0.24 -0.22 -0.2 -0.18 -0.16 -0.14 -0.12 -0.1 -0.08 -0.06 -0.04 -0.02 0.0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1.0 ]
  #define CROSSTALK 0.25 // Desaturates bright colors and preserves saturation in darker areas (inverted if negative). Helps avoiding almsost fluorescent colors [-1.0 -0.98 -0.96 -0.94 -0.92 -0.9 -0.88 -0.86 -0.84 -0.82 -0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.48 -0.46 -0.44 -0.42 -0.4 -0.38 -0.36 -0.34 -0.32 -0.3 -0.28 -0.26 -0.24 -0.22 -0.2 -0.18 -0.16 -0.14 -0.12 -0.1 -0.08 -0.06 -0.04 -0.02 0.0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1.0 ]


float luma(vec3 color){
	return dot(color,vec3(0.299, 0.587, 0.114));
}


#define ndeSat 7.0 //[30.0 29.0 28.0 27.0 26.0 25.0 24.0 23.0 22.0 21.0 20.0 19.0 18.0 17.0 16.0 15.0 14.0 13.0 12.0 11.0 10.0 9.0 8.0 7.0 6.0 5.0 4.0 3.0 2.0 1.0]
#define Purkinje_strength 1.0	// Simulates how the eye is unable to see colors at low light intensities. 0 = No purkinje effect at low exposures [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_R 0.4 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_G 0.7 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_B 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_Multiplier 0.1 // How much the purkinje effect increases brightness [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0 2.05 2.1 2.15 2.2 2.25 2.3 2.35 2.4 2.45 2.5 2.55 2.6 2.65 2.7 2.75 2.8 2.85 2.9 2.95 3.0 3.05 3.1 3.15 3.2 3.25 3.3 3.35 3.4 3.45 3.5 3.55 3.6 3.65 3.7 3.75 3.8 3.85 3.9 3.95 4.0 4.05 4.1 4.15 4.2 4.25 4.3 4.35 4.4 4.45 4.5 4.55 4.6 4.65 4.7 4.75 4.8 4.85 4.9 4.95 5.0 5.05 5.1 5.15 5.2 5.25 5.3 5.35 5.4 5.45 5.5 5.55 5.6 5.65 5.7 5.75 5.8 5.85 5.9 5.95 6.0 6.05 6.1 6.15 6.2 6.25 6.3 6.35 6.4 6.45 6.5 6.55 6.6 6.65 6.7 6.75 6.8 6.85 6.9 6.95 7.0 7.05 7.1 7.15 7.2 7.25 7.3 7.35 7.4 7.45 7.5 7.55 7.6 7.65 7.7 7.75 7.8 7.85 7.9 7.95 8.0 8.05 8.1 8.15 8.2 8.25 8.3 8.35 8.4 8.45 8.5 8.55 8.6 8.65 8.7 8.75 8.8 8.85 8.9 8.95 9.0 9.05 9.1 9.15 9.2 9.25 9.3 9.35 9.4 9.45 9.5 9.55 9.6 9.65 9.7 9.75 9.8 9.85 9.9 9.95 ]


void getNightDesaturation(inout vec3 color, float lmx) {
	float lum = dot(color,vec3(0.15,0.3,0.55));
	float lum2 = dot(color,vec3(0.85,0.7,0.45))/2;
	float rodLum = lum2*300.0;
	float rodCurve = mix(1.0, rodLum/(2.5+rodLum), 1*(Purkinje_strength));
	color = mix(lum*lmx*vec3(Purkinje_R, Purkinje_G, Purkinje_B), color, rodCurve);

	float brightness = dot(color, vec3(0.2627, 0.6780, 0.0593));
	float amount = clamp(0.1 / (pow(brightness * ndeSat, 2.0) + 0.02),0,1);
	vec3 desatColor = mix(color, vec3(brightness), vec3(0.9)) * vec3(0.2, 1.0, 2.0);

	color = mix(color, desatColor, amount);


}



#define TONEMAP_TOE_STRENGTH    0 // [-1 -0.99 -0.98 -0.97 -0.96 -0.95 -0.94 -0.93 -0.92 -0.91 -0.9 -0.89 -0.88 -0.87 -0.86 -0.85 -0.84 -0.83 -0.82 -0.81 -0.8 -0.79 -0.78 -0.77 -0.76 -0.75 -0.74 -0.73 -0.72 -0.71 -0.7 -0.69 -0.68 -0.67 -0.66 -0.64 -0.63 -0.62 -0.61 -0.6 -0.59 -0.58 -0.57 -0.56 -0.55 -0.54 -0.53 -0.52 -0.51 -0.5 -0.49 -0.48 -0.47 -0.46 -0.45 -0.44 -0.43 -0.42 -0.41 -0.4 -0.39 -0.38 -0.37 -0.36 -0.35 -0.34 -0.33 -0.32 -0.31 -0.3 -0.29 -0.28 -0.27 -0.26 -0.25 -0.24 -0.23 -0.22 -0.21 -0.2 -0.19 -0.18 -0.17 -0.16 -0.15 -0.14 -0.13 -0.12 -0.11 -0.1 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_TOE_LENGTH      0 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_LINEAR_SLOPE    1   // Should usually be left at 1
#define TONEMAP_LINEAR_LENGTH   0.5 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_SHOULDER_CURVE  0.6 // [0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1]
#define TONEMAP_SHOULDER_LENGTH 1   // Not currently in an actually useful state

#define WHITE_BALANCE 6500 // [2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000]

#define CONTRAST -0.3 // [-1 -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
#define CONTRAST_MIDPOINT 0.14





float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y));
}
vec3 int8Dither(vec3 color){
	float dither = interleaved_gradientNoise();
	return color + dither*exp2(-8.0);
}

void BSLTonemap(inout vec3 color){
	color = EXPOSURE * color;
	color = color / pow(pow(color, vec3(TONEMAP_WHITE_CURVE)) + 1.0, vec3(1.0 / TONEMAP_WHITE_CURVE));
	color = pow(color, mix(vec3(TONEMAP_LOWER_CURVE), vec3(TONEMAP_UPPER_CURVE), sqrt(color)));
}
vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

#define SAMPLE_OFFSET 5.

#define INTENSITY 0.1
float cdist(vec2 coord) {
	return max(abs(coord.s-0.5),abs(coord.t-0.5))*2.0;
}
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel,0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);    
    vec2 q0 = (q+1.0)/2.0;
    vec2 q1 = q/2.0;	
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);    
	return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}
vec4 textureQuadratic( in sampler2D sam, in vec2 p )
{
    vec2 texSize = (textureSize(sam,0).xy); 
    

    //Roger/iq style
	p = p*texSize;
	vec2 i = floor(p);
	vec2 f = fract(p);
	p = i + f*0.5;
	p = p/texSize;
    f = f*f*(3.0-2.0*f); // optional for extra sweet
	vec2 w = 0.5/texSize;
	return mix(mix(texture(sam,p+vec2(0,0)),
                   texture(sam,p+vec2(w.x,0)),f.x),
               mix(texture(sam,p+vec2(0,w.y)),
                   texture(sam,p+vec2(w.x,w.y)),f.x), f.y);
    /*

    // paniq style (https://www.shadertoy.com/view/wtXXDl)
    vec2 f = fract(p*texSize);
    vec2 c = (f*(f-1.0)+0.5) / texSize;
    vec2 w0 = p - c;
    vec2 w1 = p + c;
    return (texture(sam, vec2(w0.x, w0.y))+
    	    texture(sam, vec2(w0.x, w1.y))+
    	    texture(sam, vec2(w1.x, w1.y))+
    	    texture(sam, vec2(w1.x, w0.y)))/4.0;
#endif    
*/
    
}
void main() {

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 oneTexel = 1/ScreenSize;
    vec2 lmtrans = unpackUnorm2x4((texture(MainSampler, texCoord).a));
    vec2 lmtrans3 = unpackUnorm2x4((texture(MainSampler, texCoord+oneTexel.y).a));

          

    float lmy = mix(lmtrans.y,lmtrans3.y,res);
    float lmx = mix(lmtrans3.y,lmtrans.y,res);

    float depth = texture(DiffuseDepthSampler, texCoord).r;


    vec3 color = texture(DiffuseSampler, texCoord).rgb;


    vec2 uv = gl_FragCoord.xy / ScreenSize.xy/2. +.25;

    float vignette = (1.5-dot(texCoord-0.5,texCoord-0.5)*2.);

    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x; // make this animated
    
    vec3 img = texture( DiffuseSampler, uv*2.-.5).rgb;
      
    vec3 col = texture( blursampler, uv + vec2( i, i ) / ScreenSize ).rgb / 6.0;
  
    col += texture( blursampler, uv + vec2( i, -i ) / ScreenSize ).rgb / 6.0;
    col += texture( blursampler, uv + vec2( -i, i ) / ScreenSize ).rgb / 6.0;
    col += texture( blursampler, uv + vec2( -i, -i ) / ScreenSize ).rgb / 6.0;
    
    col += texture( blursampler, uv + vec2( 0    , i*2.0 ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( i*2. , 0     ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( -i*2., 0     ) / ScreenSize ).rgb / 12.0;
    col += texture( blursampler, uv + vec2( 0    , -i*2. ) / ScreenSize ).rgb / 12.0;
         col *= col;
    vec3 fin = max(vec3(0.0), col - 0.03);
    

	float lightScat = clamp(5.0*0.05*pow(1,0.2),0.0,1.0)*vignette;

    float VL_abs =  texture(BloomSampler, texCoord).a;
	float purkinje = 1/(1.0+1)*Purkinje_strength;
    VL_abs = clamp((1.0-VL_abs)*1.0*0.75*(1.0-purkinje),0.0,1.0)*clamp(1.0-pow(cdist(texCoord.xy),15.0),0.0,1.0);
	color = (mix(color*1.5,col,VL_abs)+fin*lightScat);
//         lmx *= clamp(pow(depth,512)*10,0,1);
	getNightDesaturation(color.rgb,clamp((lmx+lmy),0.0,5));	




	BSLTonemap(color);
    float lumC = luma(color);
	vec3 diff = color-lumC;
	color = color + diff*(-lumC*CROSSTALK + SATURATION);
  //  color.rgb = vec3(VL_abs);


	fragColor= vec4(int8Dither(vec3(color)), 1.0);
    
}
