#version 150

in vec2 texCoord;
in vec3 sunDir;
in mat4 projMat;
in mat4 modelViewMat;
in vec3 chunkOffset;
in vec3 rayDir;
in float near;
in float far;
in mat4 projInv;

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D DataSampler;
uniform sampler2D DataDepthSampler;
uniform float Time;
uniform vec2 OutSize;

out vec4 fragColor;

int imod(int val, int modulo)
{
    return val - val / modulo * modulo;
}
float interleaved_gradientNoise()
{
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y) / 1.6128);
}
ivec2 positionToPixel(vec3 position, vec2 ScreenSize, out bool inside, int discardModulo)
{
    inside = true;
    ivec2 iScreenSize = ivec2(ScreenSize);
    ivec3 iPosition = ivec3(floor(position));
    int area = iScreenSize.x * iScreenSize.y / 2;
    ivec3 sides = ivec3(int(pow(float(area), 1.0 / 3.0)));

    iPosition += sides / 2;

    if (clamp(iPosition, ivec3(0), sides - 1) != iPosition)
    {
        inside = false;
        return ivec2(-1);
    }

    int index = iPosition.x + iPosition.z * sides.x + iPosition.y * sides.x * sides.z;
    ivec2 result = ivec2(imod(index, iScreenSize.x / 2) * 2, index / (iScreenSize.x / 2) + 1);
    result.x += imod(result.y, 2);

    return result;
}

vec3 depthToView(vec2 texCoord, float depth, mat4 projInv)
{
    vec4 ndc = vec4(texCoord, depth, 1) * 2 - 1;
    vec4 viewPos = projInv * ndc;
    return viewPos.xyz / viewPos.w;
}
// simplified version of joeedh's https://www.shadertoy.com/view/Md3GWf
// see also https://www.shadertoy.com/view/MdtGD7

// --- checkerboard noise : to decorelate the pattern between size x size tiles

// simple x-y decorrelated noise seems enough
#define stepnoise0(p, size) rnd(floor(p / size) * size)
#define rnd(U) fract(sin(1e3 * (U)*mat2(1, -7.131, 12.9898, 1.233)) * 43758.5453)

//   joeedh's original noise (cleaned-up)
vec2 stepnoise(vec2 p, float size)
{
    p = floor((p + 10.) / size) * size; // is p+10. useful ?
    p = fract(p * .1) + 1. + p * vec2(2, 3) / 1e4;
    p = fract(1e5 / (.1 * p.x * (p.y + vec2(0, 1)) + 1.));
    p = fract(1e5 / (p * vec2(.1234, 2.35) + 1.));
    return p;
}

// --- stippling mask  : regular stippling + per-tile random offset +
// tone-mapping

#define SEED1 1.705
#define DMUL 8.12235325 // are exact DMUL and -.5 important ?

float mask(vec2 p)
{
    p += (stepnoise0(p, 5.5) - .5) * DMUL;                 // bias [-2,2] per tile otherwise too regular
    float f = fract(p.x * SEED1 + p.y / (SEED1 + .15555)); //  weights: 1.705 , 0.5375

    // return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the
    // image, see tests ) returned value in [0,37.2] , but < 0.57 with P=50%

    return (pow(f, 150.) + 1.3 * f) * 0.43478260869; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%
}
void main()
{
    float depth = texture(DiffuseDepthSampler, texCoord).r;
    vec3 viewPos = depthToView(texCoord, depth, projInv) * 1.0001;

    fragColor = texture(DiffuseSampler, texCoord);
    vec3 blockPos = ceil(viewPos - fract(chunkOffset));
    float noise = abs((mask(gl_FragCoord.xy + (Time * 100))) * 200);
    bool inside;
    ivec2 pixel = positionToPixel(blockPos, OutSize, inside, 0);
    /*if (inside) {
        float dataDepth = texelFetch(DataDepthSampler, pixel, 0).r;
        if (dataDepth > 0.001)
            return;
    }*/

    float shadow = 1.0;
    vec3 p = viewPos - fract(chunkOffset) + sunDir * 0.03;
    if (depth <= 0.9998)
    {
        for (int i = 0; i < 8; i++)
        {

            ivec2 pix = positionToPixel(floor(p), OutSize, inside, 0);

            if (inside && texelFetch(DataDepthSampler, pix, 0).r < 0.001)
            {
                float scale = pow(float(i) + 1, 0.5);
                shadow -= 50.0 / ((8 + noise) * scale);
                // shadow -= 1;
            }
            p += sunDir * exp(float(i + noise) / 48) * 0.5;
        }
    }
    fragColor.rgb *= max(shadow, 0.5);
}