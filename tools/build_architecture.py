"""Publication-oriented SVG model diagrams. Tensor paths follow cxr_h3_research.py."""
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'docs/figures'
BLUE, TEAL, ORANGE, INK, MUTED = '#3156a6', '#287b73', '#a96026', '#202a38', '#5f6b7a'


class Diagram:
    def __init__(self, width, height, title, subtitle):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">', '<defs>']
        for name,color in [('blue',BLUE),('teal',TEAL),('orange',ORANGE),('ink',INK)]:
            self.parts.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{color}"/></marker>')
        self.parts += ['</defs>',f'<rect width="{width}" height="{height}" fill="white"/>']
        self.text(55,60,title,34,bold=True)
        self.text(55,99,subtitle,22,color=MUTED)

    def text(self,x,y,value,size=23,color=INK,bold=False,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(value)}</text>')

    def box(self,x,y,w,h,title,lines,color=INK,fill='#f7f9fc',tag=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{color}" stroke-width="1.6"/>')
        self.text(x+20,y+36,title,min(25,(w-40)/(max(len(title),1)*.55)),color,True)
        for i,line in enumerate(lines):self.text(x+20,y+73+29*i,line,min(22,(w-40)/(max(len(line),1)*.55)))
        if tag:self.text(x+20,y+h-17,tag,min(17,(w-40)/(max(len(tag),1)*.53)),MUTED)

    def path(self,points,kind='ink',dashed=False,arrow=True):
        color={'ink':INK,'blue':BLUE,'teal':TEAL,'orange':ORANGE}[kind]
        d='M'+' L'.join(f'{x},{y}' for x,y in points)
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"'+(' stroke-dasharray="9 6"' if dashed else '')+(f' marker-end="url(#{kind})"' if arrow else '')+'/>')

    def dot(self,x,y,color=INK):self.parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
    def save(self,name):
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/name).write_text('\n'.join(self.parts+['</svg>']),encoding='utf-8')


def architecture():
    d=Diagram(1800,1190,'H3B · Conditional hierarchical variational autoencoder','128 × 128 grayscale input · global Gaussian + conditional spatial Gaussian · no deterministic encoder–decoder skips')
    d.text(55,152,'AMORTIZED INFERENCE',19,BLUE,True)
    d.box(55,185,190,135,'Input x',['1 × 128 × 128','Grayscale [0, 1]'])
    d.box(280,185,290,170,'Residual encoder',['48@64² → 80@32²','→ 128@16² → 192@8²'],BLUE,'#f2f5fb','GroupNorm · SiLU · frozen in H6A')
    d.box(625,185,260,170,'Global posterior',['qφ(z_g | x)','Pool → μ_g, log σ²_g'],BLUE,'#f2f5fb','128-D · reparameterization')
    d.box(940,185,170,170,'Global z_g',['128-D code','q(z_g | x)','or N(0, I)'],TEAL,'#f0f7f5')
    d.box(1165,185,230,170,'Global map',['Linear projection','192 × 8 × 8'],TEAL,'#f0f7f5','Frozen in H6A')
    d.path([(245,252),(280,252)],'blue',True)
    d.path([(570,252),(625,252)],'blue',True)
    d.path([(885,252),(940,252)],'blue',True)
    d.path([(1110,252),(1165,252)],'teal')
    d.text(748,392,'synthesis: z_g ~ N(0, I)',21,TEAL)
    d.path([(1025,398),(1025,355)],'teal')
    d.box(485,470,400,180,'Spatial posterior',['qφ(z_ℓ | x, z_g)','[f_32, context] → Δμ, Δlog σ²','μ_q = μ_p + Δμ'],BLUE,'#f2f5fb','80 + 96 channels · frozen in H6A')
    d.box(1050,470,345,180,'Conditional spatial prior',['pθ(z_ℓ | z_g)','96 × 32 × 32 context','μ_p, log σ²_p: 8 × 32 × 32'],TEAL,'#f0f7f5','Upsample + conv/residual · frozen in H6A')
    d.path([(425,355),(425,560),(485,560)],'blue',True)
    d.text(100,445,'f_32: 80 × 32 × 32',22,BLUE)
    d.text(100,477,'Posterior feature branch',19,MUTED)
    d.path([(1280,355),(1280,470)],'teal')
    d.path([(1050,545),(885,545)],'teal')
    d.text(965,515,'context +',18,TEAL,anchor='middle')
    d.text(965,539,'prior stats',18,TEAL,anchor='middle')
    d.box(1030,735,280,145,'Spatial z_ℓ',['8 × 32 × 32','Gaussian reparameterization'],TEAL,'#f0f7f5')
    d.path([(885,603),(945,603),(945,807),(1030,807)],'blue',True)
    d.text(790,717,'reconstruction',20,BLUE)
    d.path([(1222,650),(1222,735)],'teal')
    d.text(1244,700,'synthesis',20,TEAL)

    d.parts.append('<rect x="1450" y="419" width="295" height="633" rx="10" fill="#fffaf5" stroke="#dbc5b2"/>')
    d.text(1472,456,'RESIDUAL DECODER',20,ORANGE,True)
    d.box(1470,480,255,130,'8² → 16²',['192 → 160 channels'],ORANGE,'#fffaf5','Decoder trainable in H6A')
    d.box(1470,665,255,150,'32² injection',['160 → 128 channels','tanh(a) · Conv(z_ℓ)'],ORANGE,'#fffaf5')
    d.box(1470,870,255,130,'64² → 128²',['80 → 48 → 1 channel','Sigmoid output x̂'],ORANGE,'#fffaf5')
    d.path([(1395,252),(1597,252),(1597,480)],'ink')
    d.path([(1597,610),(1597,665)],'orange')
    d.path([(1310,807),(1390,807),(1390,740),(1470,740)],'orange')
    d.path([(1597,815),(1597,870)],'orange')
    d.dot(1280,355,TEAL)
    d.box(55,910,870,140,'Hybrid reconstruction–regularization objective',['MSE_sum + 48(1 − SSIM) + 32 Sobel-L1','+ β_t [KL_global + 0.35 KL_local]'],INK,'#f7f9fc')
    d.text(55,1099,'Solid paths: generative computation',21,TEAL)
    d.path([(510,1092),(565,1092)],'teal')
    d.text(625,1099,'Dashed paths: posterior information',21,BLUE)
    d.path([(1108,1092),(1163,1092)],'blue',True)
    d.text(55,1142,'H3B trains the full hierarchy. H6A keeps encoder/global-map/prior modules fixed and refines only the decoder.',21,MUTED)
    d.save('architecture.svg')


def refinement():
    d=Diagram(1600,700,'H6A · Restricted adversarial decoder refinement','Frozen latent distributions · trainable decoder · spectral-normalized patch critic · separate optimizer steps')
    d.box(60,190,330,170,'Frozen H3B latent paths',['Posterior: q(z_g, z_ℓ | x)','Prior: p(z_g)p(z_ℓ | z_g)','Detach sampled latent tensors'],BLUE,'#f2f5fb')
    d.box(480,190,320,170,'Trainable decoder G',['Posterior reconstruction x̂_post','Independent prior sample x̂_prior'],ORANGE,'#fffaf5','EMA 0.995 · η_G = 10⁻⁵')
    d.box(920,190,300,170,'Patch discriminator D',['Spectral-normalized convs','Real / reconstructed / prior','Hinge logits'],INK,'#f7f9fc')
    d.box(1300,190,240,170,'Real images x',['128 × 128','Brightness / contrast'],INK)
    d.path([(390,270),(480,270)],'blue')
    d.path([(800,250),(920,250)],'orange')
    d.text(860,218,'D step:',18,MUTED,anchor='middle');d.text(860,240,'detach G',18,MUTED,anchor='middle')
    d.path([(1300,270),(1220,270)],'ink')
    d.path([(1060,360),(1060,417),(640,417),(640,360)],'orange',True)
    d.text(685,451,'G step: freeze D parameters; backpropagate through D to generated pixels',21,ORANGE)
    d.box(60,505,1480,145,'Adaptive decoder objective',['L_G = L_rec + λ_t L_adv','λ_t = clip(0.05 ‖∇W L_rec‖ / (‖∇W L_adv‖ + ε), 0, 20)'],INK,'#f7f9fc')
    d.save('refinement.svg')


if __name__=='__main__':
    architecture();refinement();print('Built architecture.svg and refinement.svg')
