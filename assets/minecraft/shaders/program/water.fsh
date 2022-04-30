#version 150
#extension GL_ARB_gpu_shader5 : enable

vec4 textureGatherOffsets(sampler2D sampler, vec2 texCoord, ivec2[4] offsets, int channel)
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    return vec4(
        texelFetch(sampler, coord + offsets[0], 0)[channel], texelFetch(sampler, coord + offsets[1], 0)[channel],
        texelFetch(sampler, coord + offsets[2], 0)[channel], texelFetch(sampler, coord + offsets[3], 0)[channel]);
}

out vec4 fragColor;

uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D TerrainCloudsSampler;
uniform sampler2D temporals3Sampler;
uniform vec2 ScreenSize;
uniform float Time;
in float skys;
in vec2 texCoord;
in vec2 oneTexel;
in float skyIntensity;
in vec3 nsunColor;
in float skyIntensityNight;
in float near;
in float rainStrength;
in float far;
in float overworld;
in float end;
in vec4 fogcol;
in vec3 sunDir;
in vec3 sunPosition2;
in mat4 gbufferModelView;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 wgbufferModelViewInverse;
in vec3 ambientUp;
in vec3 ambientLeft;
in vec3 ambientRight;
in vec3 ambientB;
in vec3 ambientF;
in vec3 ambientDown;
in vec3 suncol;

#define TORCH_R 1.0
#define TORCH_G 0.7
#define TORCH_B 0.5
#define SUNBRIGHTNESS 20
float luminance(vec3 rgb)
{
    float redness = clamp(dot(rgb, vec3(1.0, -0.25, -0.75)), 0.0, 1.0);
    return ((1.0 - redness) * dot(rgb, vec3(0.2126, 0.7152, 0.0722)) + redness * 1.4) * 4.0;
}
float sqr(float x)
{
    return x * x;
}
float pow3(float x)
{
    return sqr(x) * x;
}
float pow4(float x)
{
    return sqr(x) * sqr(x);
}
float pow5(float x)
{
    return pow4(x) * x;
}
float pow6(float x)
{
    return pow5(x) * x;
}
float pow8(float x)
{
    return pow4(x) * pow4(x);
}

////////////////////////////

// moj_import doesn't work in post-process shaders ;_; Felix pls fix
#define NUMCONTROLS 26
#define THRESH 0.5
#define FPRECISION 4000000.0
#define PROJNEAR 0.05
#define FUDGE 32.0

/////////

float ld(float dist)
{
    return (2.0 * near) / (far + near - dist * (far - near));
}
vec3 nvec3(vec4 pos)
{
    return pos.xyz / pos.w;
}

vec4 nvec4(vec3 pos)
{
    return vec4(pos.xyz, 1.0);
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

#define projMAD2(m, v) (diagonal3(m) * (v) + vec3(0, 0, m[3].b))

vec3 toClipSpace3(vec3 viewSpacePosition)
{
    return projMAD2(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

#define SSPTBIAS 0.5

#define SSR_STEPS 5 //[10 15 20 25 30 35 40 50 100 200 400]

vec4 textureGood(sampler2D sam, vec2 uv)
{
    vec2 res = textureSize(sam, 0);

    vec2 st = uv * res - 0.5;

    vec2 iuv = floor(st);
    vec2 fuv = fract(st);

    vec4 a = textureLod(sam, (iuv + vec2(0.5, 0.5)) / res, 0);
    vec4 b = textureLod(sam, (iuv + vec2(1.5, 0.5)) / res, 0);
    vec4 c = textureLod(sam, (iuv + vec2(0.5, 1.5)) / res, 0);
    vec4 d = textureLod(sam, (iuv + vec2(1.5, 1.5)) / res, 0);

    return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);
}
vec3 skyLut(vec3 sVector, vec3 sunVec, float cosT, sampler2D lut)
{
    float mCosT = clamp(cosT, 0.0, 1.);
    float cosY = dot(sunVec, sVector);
    float x = ((cosY * cosY) * (cosY * 0.5 * 256.) + 0.5 * 256. + 18. + 0.5) * oneTexel.x;
    float y = (mCosT * 256. + 1.0 + 0.5) * oneTexel.y;

    return textureGood(lut, vec2(x, y)).rgb;
}
float luma(vec3 color)
{
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float pi = 3.141592;
float facos(float inX)

{

    const float C0 = 1.56467;
    const float C1 = -0.155972;

    float x = abs(inX);
    float res = C1 * x + C0;
    res *= sqrt(1.0f - x);

    return (inX >= 0) ? res : pi - res;
}
vec3 skyLut2(vec3 sVector, vec3 sunVec, float cosT, float rainStrength)
{
#define SKY_BRIGHTNESS_DAY 1.0
#define SKY_BRIGHTNESS_NIGHT 1.0;
    float mCosT = clamp(cosT, 0.0, 1.0);
    float cosY = dot(sunVec, sVector);
    float Y = facos(cosY);
    const float a = -0.8;
    const float b = -0.1;
    const float c = 3.0;
    const float d = -7.;
    const float e = 0.35;

    // luminance (cie model)
    vec3 daySky = vec3(0.0);
    vec3 moonSky = vec3(0.0);
    // Day
    if (skyIntensity > 0.00001)
    {
        float L0 = (a * exp(-0.1 / mCosT) + 1.0) * (c * (exp(d * Y) - 0.000017) + 1.0 + e * cosY * cosY);
        vec3 skyColor0 = mix(vec3(0.05, 0.5, 1.) * 0.66666, vec3(0.4, 0.5, 0.6) * 0.66666, rainStrength);
        vec3 normalizedSunColor = nsunColor;
        vec3 skyColor = mix(skyColor0, normalizedSunColor, 1.0 - pow(1.0 + L0, -1.2)) * (1.0 - rainStrength);
        daySky = pow(L0, 1.0 - rainStrength) * skyIntensity * skyColor * vec3(0.8, 0.9, 1.) * 15.0 * SKY_BRIGHTNESS_DAY;
    }
    // Night
    else if (skyIntensityNight > 0.00001)
    {

        float L0Moon = (a * exp(-0.1 / mCosT) + 1.0) * (c * (exp(d * (pi - Y)) - 0.000017) + 1.0 + e * cosY * cosY);
        moonSky = pow(L0Moon, 1.0 - rainStrength) * skyIntensityNight * vec3(0.08, 0.12, 0.18) * vec3(0.4) *
                  SKY_BRIGHTNESS_NIGHT;
    }
    return (daySky + moonSky);
}

float R2_dither()
{
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0 / 1.6180339887 * Time);
}

vec3 toScreenSpace(vec3 p)
{
    vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + vec4(0, 0, gbufferProjectionInverse[3].b, gbufferProjectionInverse[3].a);
    return fragposition.xyz / fragposition.w;
}

vec3 worldToView(vec3 worldPos)
{

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos + gbufferModelView[3];

    return pos.xyz;
}

float cubeSmooth(float x)
{
    return (x * x) * (3.0 - 2.0 * x);
}

float TextureCubic(sampler2D tex, vec2 pos)
{
    ivec2 texSize = textureSize(tex, 0) * 5;
    vec2 texelSize = (1.0 / vec2(texSize));
    float p0q0 = texture(tex, pos).a;
    float p1q0 = texture(tex, pos + vec2(texelSize.x, 0)).a;

    float p0q1 = texture(tex, pos + vec2(0, texelSize.y)).a;
    float p1q1 = texture(tex, pos + vec2(texelSize.x, texelSize.y)).a;

    float a = cubeSmooth(fract(pos.x * texSize.x));

    float pInterp_q0 = mix(p0q0, p1q0, a);
    float pInterp_q1 = mix(p0q1, p1q1, a);

    float b = cubeSmooth(fract(pos.y * texSize.y));

    return mix(pInterp_q0, pInterp_q1, b);
}
vec3 getDepthPoint(vec2 coord, float depth)
{
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; // convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;

    return pos.xyz;
}

vec3 constructNormal(float depthA, vec2 texCoords, sampler2D depthtex, float water)
{
    vec2 offsetB = vec2(0.0, oneTexel.y);
    vec2 offsetC = vec2(oneTexel.x, 0.0);
    float depthB = texture(depthtex, texCoords + offsetB).r;
    float depthC = texture(depthtex, texCoords + offsetC).r;
    vec3 A = getDepthPoint(texCoords, depthA);
    A += pow4(texture(TranslucentSampler, texCoord).g) * 0.01 * 1 * water;

    vec3 B = getDepthPoint(texCoords + offsetB, depthB);
    B += pow4(texture(TranslucentSampler, texCoord + offsetB * water).g) * 0.01 * 1 * 1;

    vec3 C = getDepthPoint(texCoords + offsetC, depthC);
    C += pow4(texture(TranslucentSampler, texCoord + offsetC * water).g) * 0.01 * 1 * 1;

    vec3 AB = normalize(B - A);
    vec3 AC = normalize(C - A);

    vec3 normal = -cross(AB, AC);
    // normal.z = -normal.z;

    return normalize(normal);
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

// --- stippling mask  : regular stippling + per-tile random offset + tone-mapping

#define SEED1 1.705
#define DMUL 8.12235325 // are exact DMUL and -.5 important ?

float mask(vec2 p)
{

    p += (stepnoise0(p, 5.5) - .5) * DMUL;                 // bias [-2,2] per tile otherwise too regular
    float f = fract(p.x * SEED1 + p.y / (SEED1 + .15555)); //  weights: 1.705 , 0.5375

    // return f;  // If you want to skeep the tone mapping
    f *= 1.03; //  to avoid zero-stipple in plain white ?

    // --- indeed, is a tone mapping ( equivalent to do the reciprocal on the image, see tests )
    // returned value in [0,37.2] , but < 0.57 with P=50%

    return (pow(f, 150.) + 1.3 * f) * 0.43478260869; // <.98 : ~ f/2, P=50%  >.98 : ~f^150, P=50%
}

vec3 rayTrace(vec3 dir, vec3 position, float dither)
{

    float quality = SSR_STEPS;
    vec3 clipPosition = nvec3(gbufferProjection * nvec4(position)) * 0.5 + 0.5;
    float rayLength =
        ((position.z + dir.z * far * 1.73205080757) > -near) ? (-near - position.z) / dir.z : far * 1.73205080757;
    vec3 direction = normalize(toClipSpace3(position + dir * rayLength) - clipPosition); // convert to clip space
    direction.xy = normalize(direction.xy);

    // get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0., direction) - clipPosition) / direction;
    float mult = min(min(maxLengths.x, maxLengths.y), maxLengths.z);

    vec3 stepv = direction * mult / quality;

    vec3 spos = clipPosition + stepv * dither;
    float minZ = clipPosition.z;
    float maxZ = spos.z + stepv.z * 0.5;
    // raymarch on a quarter res depth buffer for improved cache coherency

    for (int i = 0; i < int(quality + 1); i++)
    {

        float sp = texture2D(DiffuseDepthSampler, spos.xy).x;

        if (sp <= max(maxZ, minZ) && sp >= min(maxZ, minZ))
        {
            return vec3(spos.xy, sp);
        }
        spos += stepv;
        // small bias
        minZ = maxZ - 0.00004 / ld(spos.z);
        maxZ += stepv.z;
    }

    return vec3(1.1);
}
vec4 backProject(vec4 vec)
{
    vec4 tmp = gbufferProjectionInverse * vec;
    return tmp / tmp.w;
}

vec4 SSR(vec3 fragpos, vec3 normal, float noise)
{
    vec3 origin = backProject(vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    vec3 pos = vec3(0.0);

    vec4 color = vec4(0.0);

    vec3 reflectedVector = reflect(normalize(fragpos - origin), normalize(normal));

    pos = rayTrace(reflectedVector, fragpos, noise);

    if (pos.z < 1.0 - 1e-5)
    {

        color = texture(TerrainCloudsSampler, pos.st);
        color.rgb += texture(TranslucentSampler, pos.st).rgb * 0.45;
    }

    return color;
}

float dither5x3()
{
    const int ditherPattern[15] = int[15](9, 3, 7, 12, 0, 11, 5, 1, 14, 8, 2, 13, 10, 4, 6);

    vec2 position = floor(mod(vec2(texCoord.s * ScreenSize.x, texCoord.t * ScreenSize.y), vec2(5.0, 3.0)));

    int dither = ditherPattern[int(position.x) + int(position.y) * 5];

    return float(dither) * 0.0666666666666667f;
}
#define g(a) (-4. * a.x * a.y + 3. * a.x + a.y * 2.)
float bayer2(vec2 c)
{
    c = 0.5 * floor(c);
    return fract(1.5 * fract(c.y) + c.x);
}
float bayer16x16(vec2 p)
{

    vec2 m0 = vec2(mod(floor(p * 0.125), 2.));
    vec2 m1 = vec2(mod(floor(p * 0.25), 2.));
    vec2 m2 = vec2(mod(floor(p * 0.5), 2.));
    vec2 m3 = vec2(mod(floor(p), 2.));

    return (g(m0) + g(m1) * 4.0 + g(m2) * 16.0 + g(m3) * 64.0) * 0.003921568627451;
}
#undef g

float dither = bayer16x16(gl_FragCoord.xy);

#define bayer4(a) (bayer2(.5 * (a)) * .25 + bayer2(a))
#define bayer8(a) (bayer4(.5 * (a)) * .25 + bayer2(a))
#define bayer16(a) (bayer8(.5 * (a)) * .25 + bayer2(a))
#define bayer32(a) (bayer16(.5 * (a)) * .25 + bayer2(a))
#define bayer64(a) (bayer32(.5 * (a)) * .25 + bayer2(a))
#define bayer128(a) (bayer64(.5 * (a)) * .25 + bayer2(a))

float dither64 = bayer64(gl_FragCoord.xy);
vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    color = pow(color, vec3(0.45454545454));
    return color;
}
vec3 lumaBasedReinhardToneMapping2(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= clamp(toneMappedLuma / luma, 0, 10);
    // color = pow(color, vec3(0.45454545454));
    return color;
}
float GGX (vec3 n, vec3 v, vec3 l, float r, float F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}

vec3 viewToWorld(vec3 viewPos)
{
    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = inverse(gbufferModelView) * pos;

    return pos.xyz;
}
vec3 normVec(vec3 vec)
{
    return vec * inversesqrt(dot(vec, vec));
}



void main()
{
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

    vec4 color = texture(TranslucentSampler, texCoord);
    ivec2 texoffsets[4] = ivec2[](ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));
    vec4 OutTexel3 = (texture(TranslucentSampler, texCoord).rgba);
    vec4 cbgather = textureGatherOffsets(TranslucentSampler, texCoord, texoffsets, 2);
    vec4 crgather = textureGatherOffsets(TranslucentSampler, texCoord, texoffsets, 0);
    float lmx = clamp(mix(OutTexel3.b, dot(cbgather, vec4(1.0)) / 4, res), 0.0, 1);
    float lmy = clamp(mix(OutTexel3.r, dot(crgather, vec4(1.0)) / 4, res), 0.0, 1);
    OutTexel3.r = clamp(mix(dot(crgather, vec4(1.0)) / 4, OutTexel3.r, res), 0.0, 1);
    OutTexel3.b = clamp(mix(dot(cbgather, vec4(1.0)) / 4, OutTexel3.b, res), 0.0, 1);
    color.rgb = OutTexel3.rgb;
    vec4 color2 = color;
    float iswater = float(color.a * 255 == 200);

    float depth = texture(TranslucentDepthSampler, texCoord).r;
    float noise = mask(gl_FragCoord.xy + (Time * 100));

    float multiplier = (0.0 + ((noise+0.01) * iswater)) ;

    vec3 normal = normalize(constructNormal(depth, texCoord, TranslucentDepthSampler, multiplier));

    
            vec3 normal5 = viewToWorld(normal);

    vec3 ambientCoefs = normal / dot(abs(normal), vec3(1.0));

    vec3 ambientLight = ambientUp * clamp(ambientCoefs.y, 0., 1.);
    ambientLight += ambientDown * clamp(-ambientCoefs.y, 0., 1.);
    ambientLight += ambientRight * clamp(ambientCoefs.x, 0., 1.);
    ambientLight += ambientLeft * clamp(-ambientCoefs.x, 0., 1.);
    ambientLight += ambientB * clamp(ambientCoefs.z, 0., 1.);
    ambientLight += ambientF * clamp(-ambientCoefs.z, 0., 1.);
    ambientLight *= 1.0;
    ambientLight *= (1.0 + rainStrength * 0.2);

    ambientLight = clamp(ambientLight * (pow8(lmx) * 1.5) +
                             (pow3(lmy) * 3.0) * (vec3(TORCH_R, TORCH_G, TORCH_B) * vec3(TORCH_R, TORCH_G, TORCH_B)),
                         0.0005, 10.0);
    if (overworld != 1)
    {

        float lumC = luma(fogcol.rgb);
        vec3 diff = fogcol.rgb - lumC;

        ambientLight = clamp((diff) * (0.1) + (pow3(lmy) * 2.0) *
                                                  (vec3(TORCH_R, TORCH_G, TORCH_B) * vec3(TORCH_R, TORCH_G, TORCH_B)),
                             0.0005, 10.0);
        color.rgb = (color.rgb * ambientLight);
    }

    if (color.a > 0.01 && overworld == 1)
    {
        ////////////////////
        vec3 fragpos3 = toScreenSpace(vec3(texCoord, depth));
        vec3 screenPos2 = vec3(texCoord, depth);
        vec3 clipPos = screenPos2 * 2.0 - 1.0;
        vec4 tmp = gbufferProjectionInverse * vec4(clipPos, 1.0);
        vec3 viewPos = tmp.xyz / tmp.w;

        float normalDotEye = dot(normal, normalize(fragpos3));
        float fresnel = pow5(clamp(1.0 + normalDotEye, 0.0, 1.0));
        fresnel = fresnel * 0.98 + 0.02;
        fresnel *= 1.0 - 1 * 0.3;

        vec4 screenPos = gl_FragCoord;
        screenPos.xy = (screenPos.xy / ScreenSize - vec2(0.5)) * 2.0;
        screenPos.zw = vec2(1.0);
        vec3 view = normalize((wgbufferModelViewInverse * screenPos).xyz);
        vec3 view2 = view;
        view2.y = -view2.y;

        vec3 p3 = mat3(inverse(gbufferModelView)) * viewPos;
        vec3 view3 = normVec(p3);
        // vec3 suncol = decodeColor(texelFetch(temporals3Sampler, ivec2(8, 37), 0));

        vec3 sky_c = lumaBasedReinhardToneMapping(skyLut2(view2.xyz, sunDir, view2.y, rainStrength)) * lmx;

        vec4 reflection = vec4(sky_c.rgb, 0.);
		vec3 sunSpec = GGX(normal5,-normalize(view3),  sunPosition2, 0.1+0.05, 0.1) *suncol*(lmx-0.1);
        //sunSpec = vec3(0.0);


        reflection = vec4(SSR(viewPos.xyz, normal, noise));
        reflection.rgb = mix(sky_c.rgb, reflection.rgb, reflection.a)*1.2;
        vec3 reflected = reflection.rgb * fresnel+1*sunSpec;

        float alpha0 = color2.a;
        color.a = -color2.a * fresnel + color2.a + fresnel;


        color.rgb = clamp(
            (-0.65 * color2.rgb * alpha0 * fresnel + 0.65 * color2.rgb * alpha0 + 1.0 * reflected) / color.a, 0.0, 1.0);
        // color.rgb = vec3(normal5);
    }

    fragColor = vec4(color.rgba);
}
