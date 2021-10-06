#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

in vec2 texCoord;




 

#define SAMPLE_OFFSET 5.

#define INTENSITY 1.

void main() {
float Intensity = 1.0;
float BlurSize = 24.0;

    vec2 uv = gl_FragCoord.xy / ScreenSize.xy *2. -.5;
    vec2 res = ScreenSize.xy;
	
    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x;
    
     vec3 col = texture( DiffuseSampler, uv).rgb / 2.0;
    col += texture( DiffuseSampler, uv + vec2( i, i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( i, -i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( -i, i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( -i, -i ) / res ).rgb / 8.0;



	fragColor= vec4((col), 1.0);
//	fragColor= vec4(vec3(texture(DiffuseSampler, texCoord).a), 1.0);
}
